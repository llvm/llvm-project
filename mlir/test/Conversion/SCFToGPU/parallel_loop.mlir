// RUN: mlir-opt -convert-parallel-loops-to-gpu -split-input-file -verify-diagnostics %s | FileCheck %s

// 2-d parallel loop mapped to block.y and block.x

func.func @parallel_loop_bidy_bidx(%arg0 : index, %arg1 : index, %arg2 : index,
                              %arg3 : index, %arg4 : index,
                              %buf : memref<?x?xf32>,
                              %res : memref<?x?xf32>) {
  %step = arith.constant 2 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%arg4, %step)  {
    %val = memref.load %buf[%i0, %i1] : memref<?x?xf32>
    memref.store %val, %res[%i1, %i0] : memref<?x?xf32>
  } { mapping = [#gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>] }
  return
}

// CHECK:       #[[$MAP0:.*]] = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
// CHECK:       #[[$MAP1:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>

// CHECK:       module {
// CHECK-LABEL:   func @parallel_loop_bidy_bidx(
// CHECK-SAME:                                  [[VAL_0:%.*]]: index, [[VAL_1:%.*]]: index, [[VAL_2:%.*]]: index, [[VAL_3:%.*]]: index, [[VAL_4:%.*]]: index, [[VAL_5:%.*]]: memref<?x?xf32>, [[VAL_6:%.*]]: memref<?x?xf32>) {
// CHECK:           [[VAL_7:%.*]] = arith.constant 2 : index
// CHECK:           [[VAL_8:%.*]] = arith.constant 1 : index
// CHECK:           [[VAL_9:%.*]] = affine.apply #[[$MAP0]]([[VAL_2]]){{\[}}[[VAL_0]], [[VAL_4]]]
// CHECK:           [[VAL_10:%.*]] = affine.apply #[[$MAP0]]([[VAL_3]]){{\[}}[[VAL_1]], [[VAL_7]]]
// CHECK:           gpu.launch blocks([[VAL_11:%.*]], [[VAL_12:%.*]], [[VAL_13:%.*]]) in ([[VAL_14:%.*]] = [[VAL_10]], [[VAL_15:%.*]] = [[VAL_9]], [[VAL_16:%.*]] = [[VAL_8]]) threads([[VAL_17:%.*]], [[VAL_18:%.*]], [[VAL_19:%.*]]) in ([[VAL_20:%.*]] = [[VAL_8]], [[VAL_21:%.*]] = [[VAL_8]], [[VAL_22:%.*]] = [[VAL_8]]) {
// CHECK:             [[VAL_23:%.*]] = affine.apply #[[$MAP1]]([[VAL_12]]){{\[}}[[VAL_4]], [[VAL_0]]]
// CHECK:             [[VAL_24:%.*]] = affine.apply #[[$MAP1]]([[VAL_11]]){{\[}}[[VAL_7]], [[VAL_1]]]
// CHECK:             [[VAL_25:%.*]] = memref.load [[VAL_5]]{{\[}}[[VAL_23]], [[VAL_24]]] : memref<?x?xf32>
// CHECK:             memref.store [[VAL_25]], [[VAL_6]]{{\[}}[[VAL_24]], [[VAL_23]]] : memref<?x?xf32>
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }

// -----

// tiled 2-d parallel loop mapped to block.y and block.x and thread.y and thread.x.

func.func @parallel_loop_tiled(%arg0 : index, %arg1 : index, %arg2 : index,
                        %arg3 : index,
                        %buf : memref<?x?xf32>,
                        %res : memref<?x?xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%four, %four)  {
    scf.parallel (%si0, %si1) = (%zero, %zero) to (%four, %four)
                                            step (%one, %one)  {
      %idx0 = arith.addi %i0, %si0 : index
      %idx1 = arith.addi %i1, %si1 : index
      %val = memref.load %buf[%idx0, %idx1] : memref<?x?xf32>
      memref.store %val, %res[%idx1, %idx0] : memref<?x?xf32>
    } { mapping = [
        #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
        #gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>
     ] }
  } { mapping = [
      #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>
    ] }
  return
}

// CHECK:       #[[$MAP0:.*]] = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
// CHECK:       #[[$MAP1:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>

// CHECK:       module {
// CHECK-LABEL:   func @parallel_loop_tiled(
// CHECK-SAME:                              [[VAL_26:%.*]]: index, [[VAL_27:%.*]]: index, [[VAL_28:%.*]]: index, [[VAL_29:%.*]]: index, [[VAL_30:%.*]]: memref<?x?xf32>, [[VAL_31:%.*]]: memref<?x?xf32>) {
// CHECK:           [[VAL_32:%.*]] = arith.constant 0 : index
// CHECK:           [[VAL_33:%.*]] = arith.constant 1 : index
// CHECK:           [[VAL_34:%.*]] = arith.constant 4 : index
// CHECK:           [[VAL_35:%.*]] = arith.constant 1 : index
// CHECK:           [[VAL_36:%.*]] = affine.apply #[[$MAP0]]([[VAL_28]]){{\[}}[[VAL_26]], [[VAL_34]]]
// CHECK:           [[VAL_37:%.*]] = affine.apply #[[$MAP0]]([[VAL_29]]){{\[}}[[VAL_27]], [[VAL_34]]]
// CHECK:           [[VAL_38:%.*]] = affine.apply #[[$MAP0]]([[VAL_34]]){{\[}}[[VAL_32]], [[VAL_33]]]
// CHECK:           [[VAL_39:%.*]] = affine.apply #[[$MAP0]]([[VAL_34]]){{\[}}[[VAL_32]], [[VAL_33]]]
// CHECK:           gpu.launch blocks([[VAL_40:%.*]], [[VAL_41:%.*]], [[VAL_42:%.*]]) in ([[VAL_43:%.*]] = [[VAL_37]], [[VAL_44:%.*]] = [[VAL_36]], [[VAL_45:%.*]] = [[VAL_35]]) threads([[VAL_46:%.*]], [[VAL_47:%.*]], [[VAL_48:%.*]]) in ([[VAL_49:%.*]] = [[VAL_39]], [[VAL_50:%.*]] = [[VAL_38]], [[VAL_51:%.*]] = [[VAL_35]]) {
// CHECK:             [[VAL_52:%.*]] = affine.apply #[[$MAP1]]([[VAL_41]]){{\[}}[[VAL_34]], [[VAL_26]]]
// CHECK:             [[VAL_53:%.*]] = affine.apply #[[$MAP1]]([[VAL_40]]){{\[}}[[VAL_34]], [[VAL_27]]]
// CHECK:             [[VAL_54:%.*]] = affine.apply #[[$MAP1]]([[VAL_47]]){{\[}}[[VAL_33]], [[VAL_32]]]
// CHECK:             [[VAL_55:%.*]] = affine.apply #[[$MAP1]]([[VAL_46]]){{\[}}[[VAL_33]], [[VAL_32]]]
// CHECK:             [[VAL_56:%.*]] = arith.addi [[VAL_52]], [[VAL_54]] : index
// CHECK:             [[VAL_57:%.*]] = arith.addi [[VAL_53]], [[VAL_55]] : index
// CHECK:             [[VAL_58:%.*]] = memref.load [[VAL_30]]{{\[}}[[VAL_56]], [[VAL_57]]] : memref<?x?xf32>
// CHECK:             memref.store [[VAL_58]], [[VAL_31]]{{\[}}[[VAL_57]], [[VAL_56]]] : memref<?x?xf32>
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }

// -----

// 2-d parallel loop mapped to block.y and sequential

func.func @parallel_loop_bidy_seq(%arg0 : index, %arg1 : index, %arg2 : index,
                             %arg3 : index, %arg4 : index,
                             %buf : memref<?x?xf32>,
                             %res : memref<?x?xf32>) {
  %step = arith.constant 2 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%arg4, %step)  {
    %val = memref.load %buf[%i0, %i1] : memref<?x?xf32>
    memref.store %val, %res[%i1, %i0] : memref<?x?xf32>
  } { mapping = [
      #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>
    ] }
  return
}

// CHECK:       #[[$MAP0:.*]] = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
// CHECK:       #[[$MAP1:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>

// CHECK:       module {
// CHECK-LABEL:   func @parallel_loop_bidy_seq(
// CHECK-SAME:                                 [[VAL_59:%.*]]: index, [[VAL_60:%.*]]: index, [[VAL_61:%.*]]: index, [[VAL_62:%.*]]: index, [[VAL_63:%.*]]: index, [[VAL_64:%.*]]: memref<?x?xf32>, [[VAL_65:%.*]]: memref<?x?xf32>) {
// CHECK:           [[VAL_66:%.*]] = arith.constant 2 : index
// CHECK:           [[VAL_67:%.*]] = arith.constant 1 : index
// CHECK:           [[VAL_68:%.*]] = affine.apply #[[$MAP0]]([[VAL_61]]){{\[}}[[VAL_59]], [[VAL_63]]]
// CHECK:           gpu.launch blocks([[VAL_69:%.*]], [[VAL_70:%.*]], [[VAL_71:%.*]]) in ([[VAL_72:%.*]] = [[VAL_67]], [[VAL_73:%.*]] = [[VAL_68]], [[VAL_74:%.*]] = [[VAL_67]]) threads([[VAL_75:%.*]], [[VAL_76:%.*]], [[VAL_77:%.*]]) in ([[VAL_78:%.*]] = [[VAL_67]], [[VAL_79:%.*]] = [[VAL_67]], [[VAL_80:%.*]] = [[VAL_67]]) {
// CHECK:             [[VAL_81:%.*]] = affine.apply #[[$MAP1]]([[VAL_70]]){{\[}}[[VAL_63]], [[VAL_59]]]
// CHECK:             scf.for [[VAL_82:%.*]] = [[VAL_60]] to [[VAL_62]] step [[VAL_66]] {
// CHECK:               [[VAL_83:%.*]] = memref.load [[VAL_64]]{{\[}}[[VAL_81]], [[VAL_82]]] : memref<?x?xf32>
// CHECK:               memref.store [[VAL_83]], [[VAL_65]]{{\[}}[[VAL_82]], [[VAL_81]]] : memref<?x?xf32>
// CHECK:             }
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }

// -----

// tiled 2-d parallel loop mapped to block.y and seq. and thread.y and seq.

func.func @parallel_loop_tiled_seq(%arg0 : index, %arg1 : index, %arg2 : index,
                              %arg3 : index,
                              %buf : memref<?x?xf32>,
                              %res : memref<?x?xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%four, %four)  {
    scf.parallel (%si0, %si1) = (%zero, %zero) to (%four, %four)
                                            step (%one, %one)  {
      %idx0 = arith.addi %i0, %si0 : index
      %idx1 = arith.addi %i1, %si1 : index
      %val = memref.load %buf[%idx0, %idx1] : memref<?x?xf32>
      memref.store %val, %res[%idx1, %idx0] : memref<?x?xf32>
    } { mapping = [
        #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
        #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>
      ] }
  } { mapping = [
      #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>
    ] }
  return
}

// CHECK:       #[[$MAP0:.*]] = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
// CHECK:       #[[$MAP1:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>

// CHECK:       module {
// CHECK-LABEL:   func @parallel_loop_tiled_seq(
// CHECK-SAME:                                  [[VAL_84:%.*]]: index, [[VAL_85:%.*]]: index, [[VAL_86:%.*]]: index, [[VAL_87:%.*]]: index, [[VAL_88:%.*]]: memref<?x?xf32>, [[VAL_89:%.*]]: memref<?x?xf32>) {
// CHECK:           [[VAL_90:%.*]] = arith.constant 0 : index
// CHECK:           [[VAL_91:%.*]] = arith.constant 1 : index
// CHECK:           [[VAL_92:%.*]] = arith.constant 4 : index
// CHECK:           [[VAL_93:%.*]] = arith.constant 1 : index
// CHECK:           [[VAL_94:%.*]] = affine.apply #[[$MAP0]]([[VAL_86]]){{\[}}[[VAL_84]], [[VAL_92]]]
// CHECK:           [[VAL_95:%.*]] = affine.apply #[[$MAP0]]([[VAL_92]]){{\[}}[[VAL_90]], [[VAL_91]]]
// CHECK:           gpu.launch blocks([[VAL_96:%.*]], [[VAL_97:%.*]], [[VAL_98:%.*]]) in ([[VAL_99:%.*]] = [[VAL_93]], [[VAL_100:%.*]] = [[VAL_94]], [[VAL_101:%.*]] = [[VAL_93]]) threads([[VAL_102:%.*]], [[VAL_103:%.*]], [[VAL_104:%.*]]) in ([[VAL_105:%.*]] = [[VAL_93]], [[VAL_106:%.*]] = [[VAL_95]], [[VAL_107:%.*]] = [[VAL_93]]) {
// CHECK:             [[VAL_108:%.*]] = affine.apply #[[$MAP1]]([[VAL_97]]){{\[}}[[VAL_92]], [[VAL_84]]]
// CHECK:             scf.for [[VAL_109:%.*]] = [[VAL_85]] to [[VAL_87]] step [[VAL_92]] {
// CHECK:               [[VAL_110:%.*]] = affine.apply #[[$MAP1]]([[VAL_103]]){{\[}}[[VAL_91]], [[VAL_90]]]
// CHECK:               scf.for [[VAL_111:%.*]] = [[VAL_90]] to [[VAL_92]] step [[VAL_91]] {
// CHECK:                 [[VAL_112:%.*]] = arith.addi [[VAL_108]], [[VAL_110]] : index
// CHECK:                 [[VAL_113:%.*]] = arith.addi [[VAL_109]], [[VAL_111]] : index
// CHECK:                 [[VAL_114:%.*]] = memref.load [[VAL_88]]{{\[}}[[VAL_112]], [[VAL_113]]] : memref<?x?xf32>
// CHECK:                 memref.store [[VAL_114]], [[VAL_89]]{{\[}}[[VAL_113]], [[VAL_112]]] : memref<?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }

// -----

#map1 = affine_map<(d0)[s0] -> (2, -d0 + s0)>
#map2 = affine_map<(d0)[s0] -> (3, -d0 + s0)>

module {
  func.func @sum(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg1: memref<?x?xf32, strided<[?, 1], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %0 = memref.dim %arg0, %c0 : memref<?x?xf32, strided<[?, 1], offset: ?>>
    %1 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
    scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%0, %1) step (%c2, %c3) {
      %2 = memref.dim %arg0, %c0 : memref<?x?xf32, strided<[?, 1], offset: ?>>
      %3 = affine.min #map1(%arg3)[%2]
      %squared_min = arith.muli %3, %3 : index
      %4 = memref.dim %arg0, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
      %d = arith.subi %4, %arg4 : index
      %5 = arith.minsi %c3, %d : index
      %6 = memref.subview %arg0[%arg3, %arg4][%squared_min, %5][%c1, %c1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %7 = memref.dim %arg1, %c0 : memref<?x?xf32, strided<[?, 1], offset: ?>>
      %8 = affine.min #map1(%arg3)[%7]
      %9 = memref.dim %arg1, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
      %10 = affine.min #map2(%arg4)[%9]
      %11 = memref.subview %arg1[%arg3, %arg4][%8, %10][%c1, %c1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      %12 = memref.dim %arg2, %c0 : memref<?x?xf32, strided<[?, 1], offset: ?>>
      %13 = affine.min #map1(%arg3)[%12]
      %14 = memref.dim %arg2, %c1 : memref<?x?xf32, strided<[?, 1], offset: ?>>
      %15 = affine.min #map2(%arg4)[%14]
      %16 = memref.subview %arg2[%arg3, %arg4][%13, %15][%c1, %c1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      scf.parallel (%arg5, %arg6) = (%c0, %c0) to (%squared_min, %5) step (%c1, %c1) {
        %17 = memref.load %6[%arg5, %arg6] : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %18 = memref.load %11[%arg5, %arg6] : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %19 = memref.load %16[%arg5, %arg6] : memref<?x?xf32, strided<[?, ?], offset: ?>>
        %20 = arith.addf %17, %18 : f32
        memref.store %20, %16[%arg5, %arg6] : memref<?x?xf32, strided<[?, ?], offset: ?>>
        scf.reduce
      } {mapping = [#gpu.loop_dim_map<bound = (d0) -> (d0), map = (d0) -> (d0), processor = thread_x>, #gpu.loop_dim_map<bound = (d0) -> (d0), map = (d0) -> (d0), processor = thread_y>]}
      scf.reduce
    } {mapping = [#gpu.loop_dim_map<bound = (d0) -> (d0), map = (d0) -> (d0), processor = block_x>, #gpu.loop_dim_map<bound = (d0) -> (d0), map = (d0) -> (d0), processor = block_y>]}
    return
  }
}

// CHECK-DAG:       #[[$MAP1:.*]] = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
// CHECK-DAG:       #[[$MAP2:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
// CHECK-DAG:       #[[$MAP3:.*]] = affine_map<(d0)[s0] -> (2, -d0 + s0)>
// CHECK-DAG:       #[[$MAP4:.*]] = affine_map<(d0)[s0] -> (3, -d0 + s0)>

// CHECK:       module {
// CHECK-LABEL:   func @sum(
// CHECK-SAME:              [[VAL_0:%.*]]: memref<?x?xf32, strided<[?, 1], offset: ?>>, [[VAL_1:%.*]]: memref<?x?xf32, strided<[?, 1], offset: ?>>, [[VAL_2:%.*]]: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C3:.*]] = arith.constant 3 : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           [[VAL_7:%.*]] = memref.dim [[VAL_0]], %[[C0]] : memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:           [[VAL_8:%.*]] = memref.dim [[VAL_0]], %[[C1]] : memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:           [[VAL_9:%.*]] = arith.constant 1 : index
// CHECK:           [[VAL_10:%.*]] = affine.apply #[[$MAP1]]([[VAL_7]]){{\[}}%[[C0]], %[[C2]]]
// CHECK:           [[VAL_11:%.*]] = affine.apply #[[$MAP1]]([[VAL_8]]){{\[}}%[[C0]], %[[C3]]]
// CHECK:           [[VAL_12:%.*]] = arith.constant 4 : index
// CHECK:           [[VAL_13:%.*]] = affine.apply #[[$MAP1]]([[VAL_12]]){{\[}}%[[C0]], %[[C1]]]
// CHECK:           [[VAL_15:%.*]] = affine.apply #[[$MAP1]](%[[C3]]){{\[}}%[[C0]], %[[C1]]]
// CHECK:           gpu.launch blocks([[VAL_16:%.*]], [[VAL_17:%.*]], [[VAL_18:%.*]]) in ([[VAL_19:%.*]] = [[VAL_10]], [[VAL_20:%.*]] = [[VAL_11]], [[VAL_21:%.*]] = [[VAL_9]]) threads([[VAL_22:%.*]], [[VAL_23:%.*]], [[VAL_24:%.*]]) in ([[VAL_25:%.*]] = [[VAL_13]], [[VAL_26:%.*]] = [[VAL_15]], [[VAL_27:%.*]] = [[VAL_9]]) {
// CHECK:             [[VAL_28:%.*]] = affine.apply #[[$MAP2]]([[VAL_16]]){{\[}}%[[C2]], %[[C0]]]
// CHECK:             [[VAL_29:%.*]] = affine.apply #[[$MAP2]]([[VAL_17]]){{\[}}%[[C3]], %[[C0]]]
// CHECK:             [[VAL_30:%.*]] = memref.dim [[VAL_0]], %[[C0]] : memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:             [[VAL_31:%.*]] = affine.min #[[$MAP3]]([[VAL_28]]){{\[}}[[VAL_30]]]
// CHECK:             [[VAL_31_SQUARED:%.*]] = arith.muli [[VAL_31]], [[VAL_31]] : index
// CHECK:             [[VAL_32:%.*]] = memref.dim [[VAL_0]], %[[C1]] : memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:             [[VAL_D:%.*]] = arith.subi [[VAL_32]], [[VAL_29]] : index
// CHECK:             [[VAL_33:%.*]] = arith.minsi %[[C3]], [[VAL_D]] : index
// CHECK:             [[VAL_34:%.*]] = memref.subview [[VAL_0]]{{\[}}[[VAL_28]], [[VAL_29]]] {{\[}}[[VAL_31_SQUARED]], [[VAL_33]]] {{\[}}%[[C1]], %[[C1]]] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:             [[VAL_35:%.*]] = memref.dim [[VAL_1]], %[[C0]] : memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:             [[VAL_36:%.*]] = affine.min #[[$MAP3]]([[VAL_28]]){{\[}}[[VAL_35]]]
// CHECK:             [[VAL_37:%.*]] = memref.dim [[VAL_1]], %[[C1]] : memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:             [[VAL_38:%.*]] = affine.min #[[$MAP4]]([[VAL_29]]){{\[}}[[VAL_37]]]
// CHECK:             [[VAL_39:%.*]] = memref.subview [[VAL_1]]{{\[}}[[VAL_28]], [[VAL_29]]] {{\[}}[[VAL_36]], [[VAL_38]]] {{\[}}%[[C1]], %[[C1]]] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:             [[VAL_40:%.*]] = memref.dim [[VAL_2]], %[[C0]] : memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:             [[VAL_41:%.*]] = affine.min #[[$MAP3]]([[VAL_28]]){{\[}}[[VAL_40]]]
// CHECK:             [[VAL_42:%.*]] = memref.dim [[VAL_2]], %[[C1]] : memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:             [[VAL_43:%.*]] = affine.min #[[$MAP4]]([[VAL_29]]){{\[}}[[VAL_42]]]
// CHECK:             [[VAL_44:%.*]] = memref.subview [[VAL_2]]{{\[}}[[VAL_28]], [[VAL_29]]] {{\[}}[[VAL_41]], [[VAL_43]]] {{\[}}%[[C1]], %[[C1]]] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:             [[VAL_45:%.*]] = affine.apply #[[$MAP2]]([[VAL_22]]){{\[}}%[[C1]], %[[C0]]]
// CHECK:             [[VAL_46:%.*]] = arith.cmpi slt, [[VAL_45]], [[VAL_31_SQUARED]] : index
// CHECK:             scf.if [[VAL_46]] {
// CHECK:               [[VAL_47:%.*]] = affine.apply #[[$MAP2]]([[VAL_23]]){{\[}}%[[C1]], %[[C0]]]
// CHECK:               [[VAL_48:%.*]] = arith.cmpi slt, [[VAL_47]], [[VAL_33]] : index
// CHECK:               scf.if [[VAL_48]] {
// CHECK:                 [[VAL_49:%.*]] = memref.load [[VAL_34]]{{\[}}[[VAL_45]], [[VAL_47]]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:                 [[VAL_50:%.*]] = memref.load [[VAL_39]]{{\[}}[[VAL_45]], [[VAL_47]]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:                 [[VAL_51:%.*]] = memref.load [[VAL_44]]{{\[}}[[VAL_45]], [[VAL_47]]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:                 [[VAL_52:%.*]] = arith.addf [[VAL_49]], [[VAL_50]] : f32
// CHECK:                 memref.store [[VAL_52]], [[VAL_44]]{{\[}}[[VAL_45]], [[VAL_47]]] : memref<?x?xf32, strided<[?, ?], offset: ?>>
// CHECK:               }
// CHECK:             }
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }

// -----

// Optional attribute lowering test

func.func @parallel_loop_optional_attr() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i0) = (%c0) to (%c1) step (%c1) {
  } { mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>], optional_attr = 1 }
  // CHECK: optional_attr = 1
  return
}

// -----

// Mapping to the same processor twice. Cannot be mapped.

func.func @parallel_double_map(%arg0 : index, %arg1 : index, %arg2 : index,
                          %arg3 : index,
                          %buf : memref<?x?xf32>,
                          %res : memref<?x?xf32>) {
  %four = arith.constant 4 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%four, %four)  {
  } { mapping = [
      #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>
    ] }
  return
}

// CHECK-LABEL: @parallel_double_map
// CHECK: scf.parallel

// -----

// Loop with loop-variant upper bound. Cannot be mapped.

func.func @parallel_loop_loop_variant_bound(%arg0 : index, %arg1 : index, %arg2 : index,
                                       %arg3 : index,
                                       %buf : memref<?x?xf32>,
                                       %res : memref<?x?xf32>) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%four, %four)  {
    scf.parallel (%si0, %si1) = (%zero, %zero) to (%i0, %i1)
                                            step (%one, %one)  {
      %idx0 = arith.addi %i0, %si0 : index
      %idx1 = arith.addi %i1, %si1 : index
      %val = memref.load %buf[%idx0, %idx1] : memref<?x?xf32>
      memref.store %val, %res[%idx1, %idx0] : memref<?x?xf32>
    } { mapping = [
        #gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
        #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>
      ] }
  } { mapping = [
      #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>,
      #gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>
    ] }
  return
}

// CHECK-LABEL: @parallel_loop_loop_variant_bound
// CHECK: scf.parallel
// CHECK: scf.parallel

// -----

// Loop without annotations. Cannot be mapped.

func.func @parallel_no_annotations(%arg0 : index, %arg1 : index, %arg2 : index,
                              %arg3 : index,
                              %buf : memref<?x?xf32>,
                              %res : memref<?x?xf32>) {
  %four = arith.constant 4 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%four, %four)  {
  }
  return
}

// CHECK-LABEL: @parallel_no_annotations
// CHECK: scf.parallel

// -----

// CHECK-LABEL: @step_invariant
func.func @step_invariant() {
  %alloc = memref.alloc() : memref<1x1xf64>
  %alloc_0 = memref.alloc() : memref<1x1xf64>
  %alloc_1 = memref.alloc() : memref<1x1xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_2 = arith.constant 1 : index
  scf.parallel (%arg0) = (%c0) to (%c1) step (%c1_2) {
    %c0_3 = arith.constant 0 : index
    %c1_4 = arith.constant 1 : index
    %c1_5 = arith.constant 1 : index
    scf.parallel (%arg1) = (%c0_3) to (%c1_4) step (%c1_5) {
      %0 = memref.load %alloc_1[%arg0, %arg1] : memref<1x1xf64>
      %1 = memref.load %alloc_0[%arg0, %arg1] : memref<1x1xf64>
      %2 = arith.addf %0, %1 : f64
      memref.store %2, %alloc[%arg0, %arg1] : memref<1x1xf64>
      scf.reduce
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.reduce
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  memref.dealloc %alloc_1 : memref<1x1xf64>
  memref.dealloc %alloc_0 : memref<1x1xf64>
  memref.dealloc %alloc : memref<1x1xf64>
  return
}

// CHECK: %[[alloc_0:.*]] = memref.alloc() : memref<1x1xf64>
// CHECK: %[[alloc_1:.*]] = memref.alloc() : memref<1x1xf64>
// CHECK: %[[alloc_2:.*]] = memref.alloc() : memref<1x1xf64>
// CHECK: %[[map_0:.*]] = affine.apply #map({{.*}})[{{.*}}, {{.*}}]
// CHECK: %[[map_1:.*]] = affine.apply #map({{.*}})[{{.*}}, {{.*}}]
// CHECK: gpu.launch
// CHECK-SAME: blocks(%[[arg_0:.*]], %{{[^)]*}}, %{{[^)]*}}) in (%{{[^)]*}} = %[[map_0]], %{{[^)]*}} = %{{[^)]*}}, %{{[^)]*}} = %{{[^)]*}})
// CHECK-SAME: threads(%[[arg_3:.*]], %{{[^)]*}}, %{{[^)]*}}) in (%{{[^)]*}} = %[[map_1]], %{{[^)]*}} = %{{[^)]*}}, %{{[^)]*}} = %{{[^)]*}})
// CHECK: %[[dim0:.*]] = affine.apply #map1(%[[arg_0]])[{{.*}}, {{.*}}]
// CHECK: %[[dim1:.*]] = affine.apply #map1(%[[arg_3]])[{{.*}}, {{.*}}]
// CHECK: %[[lhs:.*]] = memref.load %[[alloc_2]][%[[dim0]], %[[dim1]]] : memref<1x1xf64>
// CHECK: %[[rhs:.*]] = memref.load %[[alloc_1]][%[[dim0]], %[[dim1]]] : memref<1x1xf64>
// CHECK: %[[sum:.*]] = arith.addf %[[lhs]], %[[rhs]] : f64
// CHECK: memref.store %[[sum]], %[[alloc_0]][%[[dim0]], %[[dim1]]] : memref<1x1xf64>
