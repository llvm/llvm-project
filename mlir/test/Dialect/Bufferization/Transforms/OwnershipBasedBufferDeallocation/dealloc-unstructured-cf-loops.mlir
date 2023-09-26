// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation \
// RUN:  -buffer-deallocation-simplification -split-input-file %s | FileCheck %s
// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation=private-function-dynamic-ownership=true -split-input-file %s > /dev/null

// RUN: mlir-opt %s -buffer-deallocation-pipeline --split-input-file > /dev/null

func.func private @infinite_loop(%arg0: memref<2xi32>) -> memref<2xi32> {
  cf.br ^body(%arg0 : memref<2xi32>)
^body(%arg1: memref<2xi32>):
  %alloc = memref.alloc() : memref<2xi32>
  cf.br ^body(%alloc : memref<2xi32>)
}

// CHECK-LABEL: func private @infinite_loop
//  CHECK-SAME: ([[ARG0:%.+]]: memref<2xi32>)
//       CHECK:   cf.br [[BODY:\^.+]]([[ARG0]], %false
//       CHECK: [[BODY]]([[M0:%.+]]: memref<2xi32>, [[OWN:%.+]]: i1):
//       CHECK:   [[ALLOC:%.+]] = memref.alloc(
// TODO: this extract_strided_metadata could be optimized away
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[M0]]
//       CHECK:   bufferization.dealloc ([[BASE]] :{{.*}}) if ([[OWN]])
//       CHECK:   cf.br [[BODY]]([[ALLOC]], %true

// -----

func.func private @simple_for_loop(
    %arg0: index, %arg1: index, %arg2: index,
    %arg3: memref<2xf32>, %arg4: memref<2xf32>
  ) {
  %alloc = memref.alloc() : memref<2xf32>
  "test.memref_user"(%alloc) : (memref<2xf32>) -> ()
  cf.br ^check(%arg0, %arg3 : index, memref<2xf32>)
^check(%0: index, %1: memref<2xf32>):  // 2 preds: ^bb0, ^body
  %2 = arith.cmpi slt, %0, %arg1 : index
  cf.cond_br %2, ^body, ^exit
^body:  // pred: ^check
  %3 = arith.cmpi eq, %0, %arg1 : index
  %alloc_0 = memref.alloc() : memref<2xf32>
  %4 = arith.addi %0, %arg2 : index
  cf.br ^check(%4, %alloc_0 : index, memref<2xf32>)
^exit:  // pred: ^check
  test.copy(%1, %arg4) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func private @simple_for_loop
//  CHECK-SAME: ([[ARG0:%.+]]: index, [[ARG1:%.+]]: index, [[ARG2:%.+]]: index, [[ARG3:%.+]]: memref<2xf32>, [[ARG4:%.+]]: memref<2xf32>)
//       CHECK:   [[ALLOC:%.+]] = memref.alloc(
//       CHECK:   test.memref_user
//       CHECK:   bufferization.dealloc ([[ALLOC]] :{{.*}}) if (%true
//       CHECK:   cf.br [[CHECK:\^.+]]([[ARG0]], [[ARG3]], %false
//       CHECK: [[CHECK]]({{.*}}: index, [[M0:%.+]]: memref<2xf32>, [[OWN:%.+]]: i1):
//       CHECK:   [[COND:%.+]] = arith.cmpi slt
//       CHECK:   [[BASE0:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[M0]]
//       CHECK:   [[THEN_OWN:%.+]] = arith.andi [[OWN]], [[COND]]
// TODO: the following dealloc could be simplified such that it doesn't have a retain
//       CHECK:   bufferization.dealloc ([[BASE0]] :{{.*}}) if ([[THEN_OWN]]) retain ([[ARG4]] :
//       CHECK:   [[NEG_COND:%.+]] = arith.xori [[COND]], %true
//       CHECK:   [[ELSE_OWN:%.+]] = arith.andi [[OWN]], [[NEG_COND]]
// TODO: the following dealloc could be completely optimized away
//       CHECK:   [[ELSE_UPDATED_OWN:%.+]]:2 = bufferization.dealloc ([[BASE0]] :{{.*}}) if ([[ELSE_OWN]]) retain ([[M0]], [[ARG4]] :
//       CHECK:   cf.cond_br [[COND]], [[BODY:\^.+]], [[EXIT:\^.+]]
//       CHECK: [[BODY]]:
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc(
//       CHECK:   cf.br [[CHECK]]({{.*}}, [[ALLOC1]], %true
//       CHECK: [[EXIT]]:
//       CHECK:   test.copy
//       CHECK:   [[BASE1:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[M0]]
//       CHECK:   bufferization.dealloc ([[BASE1]] :{{.*}}) if ([[ELSE_UPDATED_OWN]]#0)

// -----

func.func private @loop_nested_if_no_alloc(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2xf32>, %arg4: memref<2xf32>) {
  %alloc = memref.alloc() : memref<2xf32>
  cf.br ^loop_check(%arg0, %arg3 : index, memref<2xf32>)
^loop_check(%0: index, %1: memref<2xf32>):  // 2 preds: ^bb0, ^body
  %2 = arith.cmpi slt, %0, %arg1 : index
  cf.cond_br %2, ^if_check, ^exit
^if_check:  // pred: ^loop_check
  %3 = arith.cmpi eq, %0, %arg1 : index
  cf.cond_br %3, ^then, ^else
^then:  // pred: ^if_check
  cf.br ^join(%alloc : memref<2xf32>)
^else:  // pred: ^if_check
  cf.br ^join(%1 : memref<2xf32>)
^join(%4: memref<2xf32>):  // 2 preds: ^then, ^else
  cf.br ^body
^body:  // pred: ^join
  %5 = arith.addi %0, %arg2 : index
  cf.br ^loop_check(%5, %4 : index, memref<2xf32>)
^exit:  // pred: ^loop_check
  test.copy(%1, %arg4) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func private @loop_nested_if_no_alloc
//  CHECK-SAME: ([[ARG0:%.+]]: index, [[ARG1:%.+]]: index, [[ARG2:%.+]]: index, [[ARG3:%.+]]: memref<2xf32>, [[ARG4:%.+]]: memref<2xf32>)
//       CHECK:   [[ALLOC:%.+]] = memref.alloc(
//       CHECK:   cf.br [[LOOP_CHECK:\^.+]]([[ARG0]], [[ARG3]], %false
//       CHECK: [[LOOP_CHECK]]({{.*}}: index, [[M0:%.+]]: memref<2xf32>, [[OWN:%.+]]: i1):
//       CHECK:   [[OUTER_COND:%.+]] = arith.cmpi slt
//       CHECK:   [[BASE0:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[M0]]
//       CHECK:   [[THEN_OWN:%.+]] = arith.andi [[OWN]], [[OUTER_COND]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[THEN_UPDATED_OWN:%.+]]:3 = bufferization.dealloc ([[ALLOC]], [[BASE0]] :{{.*}}) if ([[OUTER_COND]], [[THEN_OWN]]) retain ([[M0]], [[ARG4]], [[ALLOC]] :
//       CHECK:   [[NEG_OUTER_COND0:%.+]] = arith.xori [[OUTER_COND]], %true
//       CHECK:   [[NEG_OUTER_COND1:%.+]] = arith.xori [[OUTER_COND]], %true
//       CHECK:   [[ELSE_OWN:%.+]] = arith.andi [[OWN]], [[NEG_OUTER_COND1]]
// TODO: this dealloc can be simplified such that it only deallocates [[ALLOC]] without any retained values in the list
//       CHECK:   [[ELSE_UPDATED_OWN:%.+]]:2 = bufferization.dealloc ([[ALLOC]], [[BASE0]] :{{.*}}) if ([[NEG_OUTER_COND0]], [[ELSE_OWN]]) retain ([[M0]], [[ARG4]] :
//       CHECK:   [[NEW_ALLOC_OWN:%.+]] = arith.select [[OUTER_COND]], [[THEN_UPDATED_OWN]]#0, [[ELSE_UPDATED_OWN]]#0
//       CHECK:   cf.cond_br [[OUTER_COND]], [[IF_CHECK:\^.+]], [[EXIT:\^.+]]
//       CHECK: [[IF_CHECK]]:
//       CHECK:   [[INNER_COND:%.+]] = arith.cmpi eq
//       CHECK:   [[BASE1:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[M0]]
//       CHECK:   [[THEN_OWN:%.+]] = arith.andi [[NEW_ALLOC_OWN]], [[INNER_COND]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   bufferization.dealloc ([[BASE1]] :{{.*}}) if ([[THEN_OWN]]) retain ([[ARG4]], [[ALLOC]] :
//       CHECK:   [[NEG_INNER_COND0:%.+]] = arith.xori [[INNER_COND]], %true
//       CHECK:   [[ELSE_OWN:%.+]] = arith.andi [[NEW_ALLOC_OWN]], [[NEG_INNER_COND0]]
//       CHECK:   [[NEG_INNER_COND1:%.+]] = arith.xori [[INNER_COND]], %true
// TODO: this dealloc can be entirely optimized away
//       CHECK:   bufferization.dealloc ([[BASE1]], [[ALLOC]] :{{.*}}) if ([[ELSE_OWN]], [[NEG_INNER_COND1]]) retain ([[M0]], [[ARG4]], [[ALLOC]] :
//       CHECK:   cf.cond_br [[INNER_COND]], [[THEN:\^.+]], [[ELSE:\^.+]]
//       CHECK: [[THEN]]:
//       CHECK:   cf.br [[JOIN:\^.+]]([[ALLOC]], %true
//       CHECK: [[ELSE]]:
//       CHECK:   [[BASE2:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[M0]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_OWN:%.+]]:3 = bufferization.dealloc ([[BASE2]], [[ALLOC]] :{{.*}}) if ([[NEW_ALLOC_OWN]], %true{{[0-9_]*}}) retain ([[M0]], [[ARG4]], [[ALLOC]] :
//       CHECK:   cf.br [[JOIN]]([[M0]], [[UPDATED_OWN]]#0
//       CHECK: [[JOIN]]([[A0:%.+]]: memref<2xf32>, [[C0:%.+]]: i1):
//       CHECK:   [[BASE3:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_OWN1:%.+]]:3 = bufferization.dealloc ([[ALLOC]], [[BASE3]] :{{.*}}) if (%true{{[0-9_]*}}, [[C0]]) retain ([[A0]], [[ARG4]], [[ALLOC]] :
//       CHECK:   cf.br [[BODY:\^.+]]
//       CHECK: [[BODY]]:
//       CHECK:   [[BASE4:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_OWN2:%.+]]:3 = bufferization.dealloc ([[BASE4]], [[ALLOC]] :{{.*}}) if ([[UPDATED_OWN1]]#0, %true{{[0-9_]*}}) retain ([[A0]], [[ARG4]], [[ALLOC]] :
//       CHECK:   cf.br [[LOOP_CHECK]]({{.*}}, [[A0]], [[UPDATED_OWN2]]#0
//       CHECK: [[EXIT]]:
//       CHECK:   test.copy
//       CHECK:   [[BASE5:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[M0]]
//       CHECK:   bufferization.dealloc ([[BASE5]] :{{.*}}) if ([[NEW_ALLOC_OWN]])

// -----

func.func private @loop_nested_if_alloc(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2xf32>) -> memref<2xf32> {
  %alloc = memref.alloc() : memref<2xf32>
  cf.br ^loop_check(%arg0, %arg3 : index, memref<2xf32>)
^loop_check(%0: index, %1: memref<2xf32>):  // 2 preds: ^bb0, ^body
  %2 = arith.cmpi slt, %0, %arg1 : index
  cf.cond_br %2, ^if_check, ^exit
^if_check:  // pred: ^loop_check
  %3 = arith.cmpi eq, %0, %arg1 : index
  cf.cond_br %3, ^then, ^else
^then:  // pred: ^if_check
  %alloc_0 = memref.alloc() : memref<2xf32>
  cf.br ^join(%alloc_0 : memref<2xf32>)
^else:  // pred: ^if_check
  cf.br ^join(%alloc : memref<2xf32>)
^join(%4: memref<2xf32>):  // 2 preds: ^then, ^else
  cf.br ^body
^body:  // pred: ^join
  %5 = arith.addi %0, %arg2 : index
  cf.br ^loop_check(%5, %4 : index, memref<2xf32>)
^exit:  // pred: ^loop_check
  return %1 : memref<2xf32>
}

// CHECK-LABEL: func private @loop_nested_if_alloc
//  CHECK-SAME: ({{.*}}: index, {{.*}}: index, {{.*}}: index, [[ARG3:%.+]]: memref<2xf32>)
//       CHECK:   [[ALLOC:%.+]] = memref.alloc(
//   CHECK-NOT:   bufferization.dealloc
//       CHECK:   cf.br [[LOOP_CHECK:\^.+]]({{.*}}, [[ARG3]], %false
//       CHECK: [[LOOP_CHECK]]({{.*}}: index, [[A0:%.+]]: memref<2xf32>, [[C0:%.+]]: i1):
//       CHECK:   [[COND:%.+]] = arith.cmpi slt
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
//       CHECK:   [[THEN_COND:%.+]] = arith.andi [[C0]], [[COND]]
//       CHECK:   bufferization.dealloc ([[BASE]] :{{.*}}) if ([[THEN_COND]]) retain ([[ALLOC]] :
// TODO: an optimization pass could move this deallocation into the "exit" block
//       CHECK:   [[ELSE_COND:%.+]] = arith.xori [[COND]], %true
//       CHECK:   bufferization.dealloc ([[ALLOC]] :{{.*}}) if ([[ELSE_COND]]) retain ([[A0]] :
//       CHECK:   cf.cond_br
//       CHECK: ^{{.*}}:
//   CHECK-NOT:   bufferization.dealloc
//       CHECK:   cf.cond_br
//       CHECK: ^{{.*}}:
//       CHECK:   [[ALLOC0:%.+]] = memref.alloc(
//   CHECK-NOT:   bufferization.dealloc
//       CHECK:   cf.br [[JOIN:\^.+]]([[ALLOC0]], %true
//       CHECK: ^{{.*}}:
//   CHECK-NOT:   bufferization.dealloc
//       CHECK:   cf.br [[JOIN]]([[ALLOC]], %true
//       CHECK: [[JOIN]]([[A1:%.+]]: memref<2xf32>, [[C1:%.+]]: i1):
//       CHECK:   [[BASE1:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A1]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_OWN:%.+]]:2 = bufferization.dealloc ([[ALLOC]], [[BASE1]] :{{.*}}) if (%true{{[0-9_]*}}, [[C1]]) retain ([[A1]], [[ALLOC]] :
//       CHECK:   cf.br
//       CHECK: ^{{.*}}:
//       CHECK:   [[BASE2:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A1]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_OWN1:%.+]]:2 = bufferization.dealloc ([[BASE2]], [[ALLOC]] :{{.*}}) if ([[UPDATED_OWN]]#0, %true{{[0-9_]*}}) retain ([[A1]], [[ALLOC]] :
//       CHECK:   cf.br ^bb1({{.*}}, [[A1]], [[UPDATED_OWN1]]#0
//       CHECK: ^{{.*}}:
//   CHECK-NOT:   bufferization.dealloc

// -----

func.func private @nested_loop_with_alloc(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2xf32>, %arg4: memref<2xf32>) {
  %alloc = memref.alloc() : memref<2xf32>
  "test.memref_user"(%alloc) : (memref<2xf32>) -> ()
  cf.br ^outer_loop_check(%arg0, %arg3 : index, memref<2xf32>)
^outer_loop_check(%0: index, %1: memref<2xf32>):  // 2 preds: ^bb0, ^outer_body
  %2 = arith.cmpi slt, %0, %arg1 : index
  cf.cond_br %2, ^outer_body_entry, ^exit
^outer_body_entry:  // pred: ^outer_loop_check
  cf.br ^middle_loop_check(%arg0, %1 : index, memref<2xf32>)
^middle_loop_check(%3: index, %4: memref<2xf32>):  // 2 preds: ^outer_body_entry, ^middle_body
  %5 = arith.cmpi slt, %3, %arg1 : index
  cf.cond_br %5, ^middle_body_entry, ^outer_body
^middle_body_entry:  // pred: ^middle_loop_check
  cf.br ^inner_loop_check(%arg0, %4 : index, memref<2xf32>)
^inner_loop_check(%6: index, %7: memref<2xf32>):  // 2 preds: ^middle_body_entry, ^inner_body
  %8 = arith.cmpi slt, %6, %arg1 : index
  cf.cond_br %8, ^if_check, ^middle_body
^if_check:  // pred: ^inner_loop_check
  %alloc_0 = memref.alloc() : memref<2xf32>
  "test.memref_user"(%alloc_0) : (memref<2xf32>) -> ()
  %9 = arith.cmpi eq, %0, %arg1 : index
  cf.cond_br %9, ^then, ^else
^then:  // pred: ^if_check
  %alloc_1 = memref.alloc() : memref<2xf32>
  cf.br ^join(%alloc_1 : memref<2xf32>)
^else:  // pred: ^if_check
  cf.br ^join(%7 : memref<2xf32>)
^join(%10: memref<2xf32>):  // 2 preds: ^then, ^else
  cf.br ^inner_body
^inner_body:  // pred: ^join
  %11 = arith.addi %6, %arg2 : index
  cf.br ^inner_loop_check(%11, %10 : index, memref<2xf32>)
^middle_body:  // pred: ^inner_loop_check
  %12 = arith.addi %3, %arg2 : index
  cf.br ^middle_loop_check(%12, %7 : index, memref<2xf32>)
^outer_body:  // pred: ^middle_loop_check
  %13 = arith.addi %0, %arg2 : index
  cf.br ^outer_loop_check(%13, %4 : index, memref<2xf32>)
^exit:  // pred: ^outer_loop_check
  test.copy(%1, %arg4) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-LABEL: func.func private @nested_loop_with_alloc
//  CHECK-SAME: ({{.*}}: index, {{.*}}: index, {{.*}}: index, [[ARG3:%.+]]: memref<2xf32>, [[ARG4:%.+]]: memref<2xf32>)
//       CHECK:   [[ALLOC:%.+]] = memref.alloc(
//       CHECK:   test.memref_user
//       CHECK:   bufferization.dealloc ([[ALLOC]] :{{.*}}) if (%true
//       CHECK:   cf.br [[OUTER_LOOP_CHECK:\^.+]]({{.*}}, [[ARG3]], %false
//       CHECK: [[OUTER_LOOP_CHECK]]({{.*}}: index, [[A0:%.+]]: memref<2xf32>, [[C0:%.+]]: i1):
//       CHECK:   [[COND:%.+]] = arith.cmpi slt
//       CHECK:   [[BASE:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
//       CHECK:   [[THEN_OWN:%.+]] = arith.andi [[C0]], [[COND]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_THEN_OWN:%.+]]:2 = bufferization.dealloc ([[BASE]] :{{.*}}) if ([[THEN_OWN]]) retain ([[A0]], [[ARG4]] :
//       CHECK:   [[NEG_COND:%.+]] = arith.xori [[COND]], %true
//       CHECK:   [[ELSE_OWN:%.+]] = arith.andi [[C0]], [[NEG_COND]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_ELSE_OWN:%.+]]:2 = bufferization.dealloc ([[BASE]] :{{.*}}) if ([[ELSE_OWN]]) retain ([[A0]], [[ARG4]] :
//       CHECK:   [[UPDATED_OWN:%.+]] = arith.select [[COND]], [[UPDATED_THEN_OWN]]#0, [[UPDATED_ELSE_OWN]]#0
//       CHECK:   cf.cond_br{{.*}}[[OUTER_BODY_ENTRY:\^.+]], [[EXIT:\^.+]]
//       CHECK: [[OUTER_BODY_ENTRY]]:
//       CHECK:   [[BASE0:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_OWN_0:%.+]]:2 = bufferization.dealloc ([[BASE0]] :{{.*}}) if ([[UPDATED_OWN]]) retain ([[A0]], [[ARG4]] :
//       CHECK:   cf.br [[MIDDLE_LOOP_CHECK:\^.+]]({{.*}}, [[A0]], [[UPDATED_OWN_0]]#0 :
//       CHECK: [[MIDDLE_LOOP_CHECK]]({{.*}}: index, [[A1:%.+]]: memref<2xf32>, [[C1:%.+]]: i1):
//       CHECK:   [[COND0:%.+]] = arith.cmpi slt
//       CHECK:   [[BASE1:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A1]]
//       CHECK:   [[THEN_OWN_0:%.+]] = arith.andi [[C1]], [[COND0]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_THEN_OWN_0:%.+]]:2 = bufferization.dealloc ([[BASE1]] :{{.*}}) if ([[THEN_OWN_0]]) retain ([[A1]], [[ARG4]] :
//       CHECK:   [[NEG_COND0:%.+]] = arith.xori [[COND0]], %true
//       CHECK:   [[ELSE_OWN_0:%.+]] = arith.andi [[C1]], [[NEG_COND0]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_ELSE_OWN_0:%.+]]:2 = bufferization.dealloc ([[BASE1]] :{{.*}}) if ([[ELSE_OWN_0]]) retain ([[A1]], [[ARG4]] :
//       CHECK:   [[UPDATED_OWN_1:%.+]] = arith.select [[COND0]], [[UPDATED_THEN_OWN_0]]#0, [[UPDATED_ELSE_OWN_0]]#0
//       CHECK:   cf.cond_br
//       CHECK: ^{{.*}}:
//       CHECK:   [[BASE2:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A1]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_OWN_2:%.+]]:2 = bufferization.dealloc ([[BASE2]] :{{.*}}) if ([[UPDATED_OWN_1]]) retain ([[A1]], [[ARG4]] :
//       CHECK:   cf.br [[IF_CHECK:\^.+]](%arg0, [[A1]], [[UPDATED_OWN_2]]#0
//       CHECK: [[IF_CHECK]]({{.*}}: index, [[A2:%.+]]: memref<2xf32>, [[C2:%.+]]: i1):
//       CHECK:   [[COND1:%.+]] = arith.cmpi slt
//       CHECK:   [[BASE3:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A2]]
//       CHECK:   [[THEN_OWN_1:%.+]] = arith.andi [[C2]], [[COND1]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_THEN_OWN_1:%.+]]:2 = bufferization.dealloc ([[BASE3]] :{{.*}}) if ([[THEN_OWN_1]]) retain ([[A2]], [[ARG4]] :
//       CHECK:   [[NEG_COND1:%.+]] = arith.xori [[COND1]], %true
//       CHECK:   [[ELSE_OWN_1:%.+]] = arith.andi [[C2]], [[NEG_COND1]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_ELSE_OWN_1:%.+]]:2 = bufferization.dealloc ([[BASE3]] :{{.*}}) if ([[ELSE_OWN_1]]) retain ([[A2]], [[ARG4]] :
//       CHECK:   [[UPDATED_OWN_3:%.+]] = arith.select [[COND1]], [[UPDATED_THEN_OWN_1]]#0, [[UPDATED_ELSE_OWN_1]]#0
//       CHECK:   cf.cond_br
//       CHECK: ^{{.*}}:
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc(
//       CHECK:   test.memref_user
//       CHECK:   [[COND2:%.+]] = arith.cmpi eq
//       CHECK:   [[BASE4:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A2]]
//       CHECK:   [[THEN_OWN_2:%.+]] = arith.andi [[UPDATED_OWN_3]], [[COND2]]
// NOTE: the retained memref can only be optimized away if the two memref function arguments are marked as restrict (guaranteed to not alias)
//       CHECK:   bufferization.dealloc ([[BASE4]] :{{.*}}) if ([[THEN_OWN_2]]) retain ([[ARG4]] :
// TODO: this dealloc could be merged with the one below by taking the disjunction of the conditions which would fold to 'true' and thus the dealloc would become unconditional
//       CHECK:   bufferization.dealloc ([[ALLOC1]] :{{.*}}) if ([[COND2]])
//       CHECK:   [[NEG_COND2:%.+]] = arith.xori [[COND2]], %true
//       CHECK:   [[ELSE_OWN_2:%.+]] = arith.andi [[UPDATED_OWN_3]], [[NEG_COND2]]
//       CHECK:   [[NEG_COND2_1:%.+]] = arith.xori [[COND2]], %true
//       CHECK:   bufferization.dealloc ([[ALLOC1]] :{{.*}}) if ([[NEG_COND2_1]])
// TODO: this dealloc can be entirely optimized away
//       CHECK:   bufferization.dealloc ([[BASE4]] :{{.*}}) if ([[ELSE_OWN_2]]) retain ([[A2]], [[ARG4]] :
//       CHECK:   cf.cond_br
//       CHECK: ^{{.*}}:
//       CHECK:   [[ALLOC2:%.+]] = memref.alloc(
//       CHECK:   cf.br [[JOIN:\^.+]]([[ALLOC2]], %true
//       CHECK: ^{{.*}}:
//       CHECK:   [[BASE5:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A2]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_OWN_4:%.+]]:2 = bufferization.dealloc ([[BASE5]] :{{.*}}) if ([[UPDATED_OWN_3]]) retain ([[A2]], [[ARG4]] :
//       CHECK:   cf.br [[JOIN]]([[A2]], [[UPDATED_OWN_4]]#0
//       CHECK: [[JOIN]]([[A3:%.+]]: memref<2xf32>, [[C3:%.+]]: i1):
//       CHECK:   [[BASE6:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A3]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_OWN_5:%.+]]:2 = bufferization.dealloc ([[BASE6]] :{{.*}}) if ([[C3]]) retain ([[A3]], [[ARG4]] :
//       CHECK:   cf.br
//       CHECK: ^{{.*}}:
//       CHECK:   [[BASE7:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A3]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_OWN_6:%.+]]:2 = bufferization.dealloc ([[BASE7]] :{{.*}}) if ([[UPDATED_OWN_5]]#0) retain ([[A3]], [[ARG4]] :
//       CHECK:   cf.br ^bb5({{.*}}, [[A3]], [[UPDATED_OWN_6]]#0
//       CHECK: ^{{.*}}:
//       CHECK:   [[BASE8:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A2]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_OWN_7:%.+]]:2 = bufferization.dealloc ([[BASE8]] :{{.*}}) if ([[UPDATED_OWN_3]]) retain ([[A2]], [[ARG4]] :
//       CHECK:   cf.br ^bb3({{.*}}, [[A2]], [[UPDATED_OWN_7]]#0
//       CHECK: ^{{.*}}:
//       CHECK:   [[BASE9:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A1]]
// TODO: this dealloc can be entirely optimized away
//       CHECK:   [[UPDATED_OWN_8:%.+]]:2 = bufferization.dealloc ([[BASE9]] :{{.*}}) if ([[UPDATED_OWN_1]]) retain ([[A1]], [[ARG4]] :
//       CHECK:   cf.br ^bb1({{.*}}, [[A1]], [[UPDATED_OWN_8]]
//       CHECK: [[EXIT]]:
//       CHECK:   test.copy
//       CHECK:   [[BASE10:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
//       CHECK:   bufferization.dealloc ([[BASE10]] :{{.*}}) if ([[UPDATED_OWN]])

// -----

func.func private @while_loop(%arg0: index) {
  %alloc = memref.alloc(%arg0) : memref<?xf32>
  cf.br ^check(%alloc, %alloc, %alloc : memref<?xf32>, memref<?xf32>, memref<?xf32>)
^check(%0: memref<?xf32>, %1: memref<?xf32>, %2: memref<?xf32>):  // 2 preds: ^bb0, ^body
  %3 = "test.make_condition"() : () -> i1
  cf.cond_br %3, ^body(%0, %1, %2 : memref<?xf32>, memref<?xf32>, memref<?xf32>), ^exit
^body(%4: memref<?xf32>, %5: memref<?xf32>, %6: memref<?xf32>):  // pred: ^check
  %alloc_0 = memref.alloc(%arg0) : memref<?xf32>
  %alloc_1 = memref.alloc(%arg0) : memref<?xf32>
  cf.br ^check(%alloc_1, %alloc_0, %5 : memref<?xf32>, memref<?xf32>, memref<?xf32>)
^exit:  // pred: ^check
  return
}

// CHECK-LABEL: func private @while_loop
//       CHECK:   [[ALLOC:%.+]] = memref.alloc(
//       CHECK:   cf.br [[CHECK:\^.+]]([[ALLOC]], [[ALLOC]], [[ALLOC]], %true{{[0-9_]*}}, %true{{[0-9_]*}}, %true
//       CHECK: [[CHECK]]([[I0:%.+]]: memref<?xf32>, [[I1:%.+]]: memref<?xf32>, [[I2:%.+]]: memref<?xf32>, [[I3:%.+]]: i1, [[I4:%.+]]: i1, [[I5:%.+]]: i1):
//       CHECK:   [[COND:%.+]] = "test.make_condition"
//       CHECK:   [[BASE0:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[I0]]
//       CHECK:   [[BASE1:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[I1]]
//       CHECK:   [[BASE2:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[I2]]
//       CHECK:   [[OWN0:%.+]] = arith.andi [[I3]], [[COND]]
//       CHECK:   [[OWN1:%.+]] = arith.andi [[I4]], [[COND]]
//       CHECK:   [[OWN2:%.+]] = arith.andi [[I5]], [[COND]]
// TODO: this dealloc can be optimized away entirely
//       CHECK:   [[THEN_UPDATED_OWN:%.+]]:3 = bufferization.dealloc ([[BASE0]], [[BASE1]], [[BASE2]] :{{.*}}) if ([[OWN0]], [[OWN1]], [[OWN2]]) retain ([[I0]], [[I1]], [[I2]] :
//       CHECK:   [[NEG_COND:%.+]] = arith.xori [[COND]], %true
//       CHECK:   [[OWN3:%.+]] = arith.andi [[I3]], [[NEG_COND]]
//       CHECK:   [[NEG_COND:%.+]] = arith.xori [[COND]], %true
//       CHECK:   [[OWN4:%.+]] = arith.andi [[I4]], [[NEG_COND]]
//       CHECK:   [[NEG_COND:%.+]] = arith.xori [[COND]], %true
//       CHECK:   [[OWN5:%.+]] = arith.andi [[I5]], [[NEG_COND]]
// TODO: it would be good to have an optimization that moves this deallocation to the exit block instead
//       CHECK:   bufferization.dealloc ([[BASE0]], [[BASE1]], [[BASE2]] :{{.*}}) if ([[OWN3]], [[OWN4]], [[OWN5]])
//       CHECK:   cf.cond_br [[COND]], [[BODY:\^.+]]([[I0]], [[I1]], [[I2]], [[THEN_UPDATED_OWN]]#0, [[THEN_UPDATED_OWN]]#1, [[THEN_UPDATED_OWN]]#2 :{{.*}}), [[EXIT:\^.+]]
//       CHECK: [[BODY]]([[A0:%.+]]: memref<?xf32>, [[A1:%.+]]: memref<?xf32>, [[A2:%.+]]: memref<?xf32>, [[A3:%.+]]: i1, [[A4:%.+]]: i1, [[A5:%.+]]: i1):
//       CHECK:   [[ALLOC0:%.+]] = memref.alloc(
//       CHECK:   [[ALLOC1:%.+]] = memref.alloc(
//       CHECK:   [[BASE0:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A0]]
//       CHECK:   [[BASE1:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A1]]
//       CHECK:   [[BASE2:%[a-zA-Z0-9_]+]]{{.*}} = memref.extract_strided_metadata [[A2]]
// TODO: this dealloc op could be considerably simplified by some better analysis
//       CHECK:   [[UPDATED_OWN:%.+]]:3 = bufferization.dealloc ([[BASE0]], [[BASE1]], [[BASE2]], [[ALLOC0]] :{{.*}}) if ([[A3]], [[A4]], [[A5]], %true{{[0-9_]*}}) retain ([[ALLOC1]], [[ALLOC0]], [[A1]] :
//       CHECK:   cf.br [[CHECK]]([[ALLOC1]], [[ALLOC0]], [[A1]], %true{{[0-9_]*}}, %true{{[0-9_]*}}, [[UPDATED_OWN]]#2 :
//       CHECK: [[EXIT]]:
//       CHECK:   return
