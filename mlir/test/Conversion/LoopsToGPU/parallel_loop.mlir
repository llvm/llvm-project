// RUN: mlir-opt -convert-parallel-loops-to-gpu -split-input-file %s | FileCheck %s -dump-input-on-failure

// 2-d parallel loop mapped to block.y and block.x

func @parallel_loop_bidy_bidx(%arg0 : index, %arg1 : index, %arg2 : index,
                              %arg3 : index, %arg4 : index, 
                              %buf : memref<?x?xf32>,
                              %res : memref<?x?xf32>) {
  %step = constant 2 : index
  loop.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%arg4, %step)  {
    %val = load %buf[%i0, %i1] : memref<?x?xf32>
    store %val, %res[%i1, %i0] : memref<?x?xf32>
  } { mapping = [{processor = 1, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>}, {processor = 0, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>}] }
  return
}

// CHECK:       #map0 = affine_map<(d0) -> (d0)>
// CHECK:       module {

// CHECK-LABEL:   func @parallel_loop_bidy_bidx(
// CHECK-SAME:                        [[VAL_0:%.*]]: index, [[VAL_1:%.*]]: index, [[VAL_2:%.*]]: index, [[VAL_3:%.*]]: index, [[VAL_4:%.*]]: index, [[VAL_5:%.*]]: memref<?x?xf32>, [[VAL_6:%.*]]: memref<?x?xf32>) {
// CHECK:           [[VAL_7:%.*]] = constant 2 : index
// CHECK:           [[VAL_8:%.*]] = constant 1 : index
// CHECK:           [[VAL_9:%.*]] = subi [[VAL_2]], [[VAL_0]] : index
// CHECK:           [[VAL_10:%.*]] = affine.apply #map0([[VAL_9]])
// CHECK:           [[VAL_11:%.*]] = subi [[VAL_3]], [[VAL_1]] : index
// CHECK:           [[VAL_12:%.*]] = affine.apply #map0([[VAL_11]])
// CHECK:           gpu.launch blocks([[VAL_13:%.*]], [[VAL_14:%.*]], [[VAL_15:%.*]]) in ([[VAL_16:%.*]] = [[VAL_12]], [[VAL_17:%.*]] = [[VAL_10]], [[VAL_18:%.*]] = [[VAL_8]]) threads([[VAL_19:%.*]], [[VAL_20:%.*]], [[VAL_21:%.*]]) in ([[VAL_22:%.*]] = [[VAL_8]], [[VAL_23:%.*]] = [[VAL_8]], [[VAL_24:%.*]] = [[VAL_8]]) {
// CHECK:             [[VAL_25:%.*]] = affine.apply #map0([[VAL_14]])
// CHECK:             [[VAL_26:%.*]] = addi [[VAL_25]], [[VAL_0]] : index
// CHECK:             [[VAL_27:%.*]] = affine.apply #map0([[VAL_13]])
// CHECK:             [[VAL_28:%.*]] = addi [[VAL_27]], [[VAL_1]] : index
// CHECK:             [[VAL_29:%.*]] = load [[VAL_5]]{{\[}}[[VAL_26]], [[VAL_28]]] : memref<?x?xf32>
// CHECK:             store [[VAL_29]], [[VAL_6]]{{\[}}[[VAL_28]], [[VAL_26]]] : memref<?x?xf32>
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }

// -----

// tiled 2-d parallel loop mapped to block.y and block.x and thread.y and thread.x.

func @parallel_loop_tiled(%arg0 : index, %arg1 : index, %arg2 : index,
                        %arg3 : index,
                        %buf : memref<?x?xf32>,
                        %res : memref<?x?xf32>) {
  %zero = constant 0 : index
  %one = constant 1 : index
  %four = constant 4 : index
  loop.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%four, %four)  {
    loop.parallel (%si0, %si1) = (%zero, %zero) to (%four, %four)
                                            step (%one, %one)  {
      %idx0 = addi %i0, %si0 : index
      %idx1 = addi %i1, %si1 : index
      %val = load %buf[%idx0, %idx1] : memref<?x?xf32>
      store %val, %res[%idx1, %idx0] : memref<?x?xf32>
    } { mapping = [
        {processor = 4, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>},
        {processor = 3, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>}
     ] }
  } { mapping = [
      {processor = 1, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>},
      {processor = 0, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>}
    ] }
  return
}

// CHECK:       #map0 = affine_map<(d0) -> (d0)>
// CHECK:       module {

// CHECK-LABEL:   func @parallel_loop_tiled(
// CHECK-SAME:                              [[VAL_30:%.*]]: index, [[VAL_31:%.*]]: index, [[VAL_32:%.*]]: index, [[VAL_33:%.*]]: index, [[VAL_34:%.*]]: memref<?x?xf32>, [[VAL_35:%.*]]: memref<?x?xf32>) {
// CHECK:           [[VAL_36:%.*]] = constant 0 : index
// CHECK:           [[VAL_37:%.*]] = constant 1 : index
// CHECK:           [[VAL_38:%.*]] = constant 4 : index
// CHECK:           [[VAL_39:%.*]] = constant 1 : index
// CHECK:           [[VAL_40:%.*]] = subi [[VAL_32]], [[VAL_30]] : index
// CHECK:           [[VAL_41:%.*]] = affine.apply #map0([[VAL_40]])
// CHECK:           [[VAL_42:%.*]] = subi [[VAL_33]], [[VAL_31]] : index
// CHECK:           [[VAL_43:%.*]] = affine.apply #map0([[VAL_42]])
// CHECK:           [[VAL_44:%.*]] = subi [[VAL_38]], [[VAL_36]] : index
// CHECK:           [[VAL_45:%.*]] = affine.apply #map0([[VAL_44]])
// CHECK:           [[VAL_46:%.*]] = subi [[VAL_38]], [[VAL_36]] : index
// CHECK:           [[VAL_47:%.*]] = affine.apply #map0([[VAL_46]])
// CHECK:           gpu.launch blocks([[VAL_48:%.*]], [[VAL_49:%.*]], [[VAL_50:%.*]]) in ([[VAL_51:%.*]] = [[VAL_43]], [[VAL_52:%.*]] = [[VAL_41]], [[VAL_53:%.*]] = [[VAL_39]]) threads([[VAL_54:%.*]], [[VAL_55:%.*]], [[VAL_56:%.*]]) in ([[VAL_57:%.*]] = [[VAL_47]], [[VAL_58:%.*]] = [[VAL_45]], [[VAL_59:%.*]] = [[VAL_39]]) {
// CHECK:             [[VAL_60:%.*]] = affine.apply #map0([[VAL_49]])
// CHECK:             [[VAL_61:%.*]] = addi [[VAL_60]], [[VAL_30]] : index
// CHECK:             [[VAL_62:%.*]] = affine.apply #map0([[VAL_48]])
// CHECK:             [[VAL_63:%.*]] = addi [[VAL_62]], [[VAL_31]] : index
// CHECK:             [[VAL_64:%.*]] = affine.apply #map0([[VAL_55]])
// CHECK:             [[VAL_65:%.*]] = addi [[VAL_64]], [[VAL_36]] : index
// CHECK:             [[VAL_66:%.*]] = affine.apply #map0([[VAL_54]])
// CHECK:             [[VAL_67:%.*]] = addi [[VAL_66]], [[VAL_36]] : index
// CHECK:             [[VAL_68:%.*]] = addi [[VAL_61]], [[VAL_65]] : index
// CHECK:             [[VAL_69:%.*]] = addi [[VAL_63]], [[VAL_67]] : index
// CHECK:             [[VAL_70:%.*]] = load [[VAL_34]]{{\[}}[[VAL_68]], [[VAL_69]]] : memref<?x?xf32>
// CHECK:             store [[VAL_70]], [[VAL_35]]{{\[}}[[VAL_69]], [[VAL_68]]] : memref<?x?xf32>
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }

// -----

// 2-d parallel loop mapped to block.y and sequential

func @parallel_loop_bidy_seq(%arg0 : index, %arg1 : index, %arg2 : index,
                             %arg3 : index, %arg4 : index,
                             %buf : memref<?x?xf32>,
                             %res : memref<?x?xf32>) {
  %step = constant 2 : index
  loop.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%arg4, %step)  {
    %val = load %buf[%i0, %i1] : memref<?x?xf32>
    store %val, %res[%i1, %i0] : memref<?x?xf32>
  } { mapping = [
      {processor = 1, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>},
      {processor = 6, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>}
    ] }
  return
}

// CHECK:       #map0 = affine_map<(d0) -> (d0)>
// CHECK:       module {

// CHECK-LABEL:   func @parallel_loop_bidy_seq(
// CHECK-SAME:                        [[VAL_71:%.*]]: index, [[VAL_72:%.*]]: index, [[VAL_73:%.*]]: index, [[VAL_74:%.*]]: index, [[VAL_75:%.*]]: index, [[VAL_76:%.*]]: memref<?x?xf32>, [[VAL_77:%.*]]: memref<?x?xf32>) {
// CHECK:           [[VAL_78:%.*]] = constant 2 : index
// CHECK:           [[VAL_79:%.*]] = constant 1 : index
// CHECK:           [[VAL_80:%.*]] = subi [[VAL_73]], [[VAL_71]] : index
// CHECK:           [[VAL_81:%.*]] = affine.apply #map0([[VAL_80]])
// CHECK:           gpu.launch blocks([[VAL_82:%.*]], [[VAL_83:%.*]], [[VAL_84:%.*]]) in ([[VAL_85:%.*]] = [[VAL_79]], [[VAL_86:%.*]] = [[VAL_81]], [[VAL_87:%.*]] = [[VAL_79]]) threads([[VAL_88:%.*]], [[VAL_89:%.*]], [[VAL_90:%.*]]) in ([[VAL_91:%.*]] = [[VAL_79]], [[VAL_92:%.*]] = [[VAL_79]], [[VAL_93:%.*]] = [[VAL_79]]) {
// CHECK:             [[VAL_94:%.*]] = affine.apply #map0([[VAL_83]])
// CHECK:             [[VAL_95:%.*]] = addi [[VAL_94]], [[VAL_71]] : index
// CHECK:             loop.for [[VAL_96:%.*]] = [[VAL_72]] to [[VAL_74]] step [[VAL_78]] {
// CHECK:               [[VAL_97:%.*]] = load [[VAL_76]]{{\[}}[[VAL_95]], [[VAL_96]]] : memref<?x?xf32>
// CHECK:               store [[VAL_97]], [[VAL_77]]{{\[}}[[VAL_96]], [[VAL_95]]] : memref<?x?xf32>
// CHECK:             }
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }

// -----

// tiled 2-d parallel loop mapped to block.y and seq. and thread.y and seq.

func @parallel_loop_tiled_seq(%arg0 : index, %arg1 : index, %arg2 : index,
                              %arg3 : index,
                              %buf : memref<?x?xf32>,
                              %res : memref<?x?xf32>) {
  %zero = constant 0 : index
  %one = constant 1 : index
  %four = constant 4 : index
  loop.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%four, %four)  {
    loop.parallel (%si0, %si1) = (%zero, %zero) to (%four, %four)
                                            step (%one, %one)  {
      %idx0 = addi %i0, %si0 : index
      %idx1 = addi %i1, %si1 : index
      %val = load %buf[%idx0, %idx1] : memref<?x?xf32>
      store %val, %res[%idx1, %idx0] : memref<?x?xf32>
    } { mapping = [
        {processor = 4, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>},
        {processor = 6, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>}
      ] }
  } { mapping = [
      {processor = 1, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>},
      {processor = 6, map = affine_map<(d0) -> (d0)>, bound = affine_map<(d0) -> (d0)>}
    ] }
  return
}

// CHECK:       #map0 = affine_map<(d0) -> (d0)>
// CHECK:       module {

// CHECK-LABEL:   func @parallel_loop_tiled_seq(
// CHECK-SAME:                        [[VAL_98:%.*]]: index, [[VAL_99:%.*]]: index, [[VAL_100:%.*]]: index, [[VAL_101:%.*]]: index, [[VAL_102:%.*]]: memref<?x?xf32>, [[VAL_103:%.*]]: memref<?x?xf32>) {
// CHECK:           [[VAL_104:%.*]] = constant 0 : index
// CHECK:           [[VAL_105:%.*]] = constant 1 : index
// CHECK:           [[VAL_106:%.*]] = constant 4 : index
// CHECK:           [[VAL_107:%.*]] = constant 1 : index
// CHECK:           [[VAL_108:%.*]] = subi [[VAL_100]], [[VAL_98]] : index
// CHECK:           [[VAL_109:%.*]] = affine.apply #map0([[VAL_108]])
// CHECK:           [[VAL_110:%.*]] = subi [[VAL_106]], [[VAL_104]] : index
// CHECK:           [[VAL_111:%.*]] = affine.apply #map0([[VAL_110]])
// CHECK:           gpu.launch blocks([[VAL_112:%.*]], [[VAL_113:%.*]], [[VAL_114:%.*]]) in ([[VAL_115:%.*]] = [[VAL_107]], [[VAL_116:%.*]] = [[VAL_109]], [[VAL_117:%.*]] = [[VAL_107]]) threads([[VAL_118:%.*]], [[VAL_119:%.*]], [[VAL_120:%.*]]) in ([[VAL_121:%.*]] = [[VAL_107]], [[VAL_122:%.*]] = [[VAL_111]], [[VAL_123:%.*]] = [[VAL_107]]) {
// CHECK:             [[VAL_124:%.*]] = affine.apply #map0([[VAL_113]])
// CHECK:             [[VAL_125:%.*]] = addi [[VAL_124]], [[VAL_98]] : index
// CHECK:             loop.for [[VAL_126:%.*]] = [[VAL_99]] to [[VAL_101]] step [[VAL_106]] {
// CHECK:               [[VAL_127:%.*]] = affine.apply #map0([[VAL_119]])
// CHECK:               [[VAL_128:%.*]] = addi [[VAL_127]], [[VAL_104]] : index
// CHECK:               loop.for [[VAL_129:%.*]] = [[VAL_104]] to [[VAL_106]] step [[VAL_105]] {
// CHECK:                 [[VAL_130:%.*]] = addi [[VAL_125]], [[VAL_128]] : index
// CHECK:                 [[VAL_131:%.*]] = addi [[VAL_126]], [[VAL_129]] : index
// CHECK:                 [[VAL_132:%.*]] = load [[VAL_102]]{{\[}}[[VAL_130]], [[VAL_131]]] : memref<?x?xf32>
// CHECK:                 store [[VAL_132]], [[VAL_103]]{{\[}}[[VAL_131]], [[VAL_130]]] : memref<?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }

// -----

#map0 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
#map2 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#map3 = affine_map<(d0) -> (d0)>

module {
  func @sum(%arg0: memref<?x?xf32, #map0>, %arg1: memref<?x?xf32, #map0>, %arg2: memref<?x?xf32, #map0>) {
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %c3 = constant 3 : index
    %c2 = constant 2 : index
    %0 = dim %arg0, 0 : memref<?x?xf32, #map0>
    %1 = dim %arg0, 1 : memref<?x?xf32, #map0>
    loop.parallel (%arg3, %arg4) = (%c0, %c0) to (%0, %1) step (%c2, %c3) {
      %2 = dim %arg0, 0 : memref<?x?xf32, #map0>
      %3 = affine.min #map1(%c2, %2, %arg3)
      %4 = dim %arg0, 1 : memref<?x?xf32, #map0>
      %5 = affine.min #map1(%c3, %4, %arg4)
      %6 = std.subview %arg0[%arg3, %arg4][%3, %5][%c1, %c1] : memref<?x?xf32, #map0> to memref<?x?xf32, #map2>
      %7 = dim %arg1, 0 : memref<?x?xf32, #map0>
      %8 = affine.min #map1(%c2, %7, %arg3)
      %9 = dim %arg1, 1 : memref<?x?xf32, #map0>
      %10 = affine.min #map1(%c3, %9, %arg4)
      %11 = std.subview %arg1[%arg3, %arg4][%8, %10][%c1, %c1] : memref<?x?xf32, #map0> to memref<?x?xf32, #map2>
      %12 = dim %arg2, 0 : memref<?x?xf32, #map0>
      %13 = affine.min #map1(%c2, %12, %arg3)
      %14 = dim %arg2, 1 : memref<?x?xf32, #map0>
      %15 = affine.min #map1(%c3, %14, %arg4)
      %16 = std.subview %arg2[%arg3, %arg4][%13, %15][%c1, %c1] : memref<?x?xf32, #map0> to memref<?x?xf32, #map2>
      loop.parallel (%arg5, %arg6) = (%c0, %c0) to (%3, %5) step (%c1, %c1) {
        %17 = load %6[%arg5, %arg6] : memref<?x?xf32, #map2>
        %18 = load %11[%arg5, %arg6] : memref<?x?xf32, #map2>
        %19 = load %16[%arg5, %arg6] : memref<?x?xf32, #map2>
        %20 = addf %17, %18 : f32
        store %20, %16[%arg5, %arg6] : memref<?x?xf32, #map2>
        "loop.terminator"() : () -> ()
      } { mapping = [
          {processor = 3, map = #map3, bound = #map3},
          {processor = 4, map = #map3, bound = #map3}
        ] }
      "loop.terminator"() : () -> ()
    } { mapping = [
        {processor = 0, map = #map3, bound = #map3},
        {processor = 1, map = #map3, bound = #map3}
    ] }
    return
  }
}

// CHECK:       #map0 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK:       #map1 = affine_map<(d0) -> (d0)>
// CHECK:       #map2 = affine_map<(d0, d1, d2) -> (d0, d1 - d2)>
// CHECK:       #map3 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
// CHECK:       module {

// CHECK-LABEL:   func @sum(
// CHECK-SAME:              [[VAL_133:%.*]]: memref<?x?xf32, #map0>, [[VAL_134:%.*]]: memref<?x?xf32, #map0>, [[VAL_135:%.*]]: memref<?x?xf32, #map0>) {
// CHECK:           [[VAL_136:%.*]] = constant 1 : index
// CHECK:           [[VAL_137:%.*]] = constant 0 : index
// CHECK:           [[VAL_138:%.*]] = constant 3 : index
// CHECK:           [[VAL_139:%.*]] = constant 2 : index
// CHECK:           [[VAL_140:%.*]] = dim [[VAL_133]], 0 : memref<?x?xf32, #map0>
// CHECK:           [[VAL_141:%.*]] = dim [[VAL_133]], 1 : memref<?x?xf32, #map0>
// CHECK:           [[VAL_142:%.*]] = constant 1 : index
// CHECK:           [[VAL_143:%.*]] = subi [[VAL_140]], [[VAL_137]] : index
// CHECK:           [[VAL_144:%.*]] = affine.apply #map1([[VAL_143]])
// CHECK:           [[VAL_145:%.*]] = subi [[VAL_141]], [[VAL_137]] : index
// CHECK:           [[VAL_146:%.*]] = affine.apply #map1([[VAL_145]])
// CHECK:           [[VAL_148:%.*]] = subi [[VAL_139]], [[VAL_137]] : index
// CHECK:           [[VAL_149:%.*]] = affine.apply #map1([[VAL_148]])
// CHECK:           [[VAL_151:%.*]] = subi [[VAL_138]], [[VAL_137]] : index
// CHECK:           [[VAL_152:%.*]] = affine.apply #map1([[VAL_151]])
// CHECK:           gpu.launch blocks([[VAL_153:%.*]], [[VAL_154:%.*]], [[VAL_155:%.*]]) in ([[VAL_156:%.*]] = [[VAL_144]], [[VAL_157:%.*]] = [[VAL_146]], [[VAL_158:%.*]] = [[VAL_142]]) threads([[VAL_159:%.*]], [[VAL_160:%.*]], [[VAL_161:%.*]]) in ([[VAL_162:%.*]] = [[VAL_149]], [[VAL_163:%.*]] = [[VAL_152]], [[VAL_164:%.*]] = [[VAL_142]]) {
// CHECK:             [[VAL_165:%.*]] = affine.apply #map1([[VAL_153]])
// CHECK:             [[VAL_166:%.*]] = addi [[VAL_165]], [[VAL_137]] : index
// CHECK:             [[VAL_167:%.*]] = affine.apply #map1([[VAL_154]])
// CHECK:             [[VAL_168:%.*]] = addi [[VAL_167]], [[VAL_137]] : index
// CHECK:             [[VAL_169:%.*]] = dim [[VAL_133]], 0 : memref<?x?xf32, #map0>
// CHECK:             [[VAL_170:%.*]] = affine.min #map2([[VAL_139]], [[VAL_169]], [[VAL_166]])
// CHECK:             [[VAL_171:%.*]] = dim [[VAL_133]], 1 : memref<?x?xf32, #map0>
// CHECK:             [[VAL_172:%.*]] = affine.min #map2([[VAL_138]], [[VAL_171]], [[VAL_168]])
// CHECK:             [[VAL_173:%.*]] = std.subview [[VAL_133]]{{\[}}[[VAL_166]], [[VAL_168]]]{{\[}}[[VAL_170]], [[VAL_172]]]{{\[}}[[VAL_136]], [[VAL_136]]] : memref<?x?xf32, #map0> to memref<?x?xf32, #map3>
// CHECK:             [[VAL_174:%.*]] = dim [[VAL_134]], 0 : memref<?x?xf32, #map0>
// CHECK:             [[VAL_175:%.*]] = affine.min #map2([[VAL_139]], [[VAL_174]], [[VAL_166]])
// CHECK:             [[VAL_176:%.*]] = dim [[VAL_134]], 1 : memref<?x?xf32, #map0>
// CHECK:             [[VAL_177:%.*]] = affine.min #map2([[VAL_138]], [[VAL_176]], [[VAL_168]])
// CHECK:             [[VAL_178:%.*]] = std.subview [[VAL_134]]{{\[}}[[VAL_166]], [[VAL_168]]]{{\[}}[[VAL_175]], [[VAL_177]]]{{\[}}[[VAL_136]], [[VAL_136]]] : memref<?x?xf32, #map0> to memref<?x?xf32, #map3>
// CHECK:             [[VAL_179:%.*]] = dim [[VAL_135]], 0 : memref<?x?xf32, #map0>
// CHECK:             [[VAL_180:%.*]] = affine.min #map2([[VAL_139]], [[VAL_179]], [[VAL_166]])
// CHECK:             [[VAL_181:%.*]] = dim [[VAL_135]], 1 : memref<?x?xf32, #map0>
// CHECK:             [[VAL_182:%.*]] = affine.min #map2([[VAL_138]], [[VAL_181]], [[VAL_168]])
// CHECK:             [[VAL_183:%.*]] = std.subview [[VAL_135]]{{\[}}[[VAL_166]], [[VAL_168]]]{{\[}}[[VAL_180]], [[VAL_182]]]{{\[}}[[VAL_136]], [[VAL_136]]] : memref<?x?xf32, #map0> to memref<?x?xf32, #map3>
// CHECK:             [[VAL_184:%.*]] = affine.apply #map1([[VAL_159]])
// CHECK:             [[VAL_185:%.*]] = addi [[VAL_184]], [[VAL_137]] : index
// CHECK:             [[VAL_186:%.*]] = cmpi "slt", [[VAL_185]], [[VAL_170]] : index
// CHECK:             loop.if [[VAL_186]] {
// CHECK:               [[VAL_187:%.*]] = affine.apply #map1([[VAL_160]])
// CHECK:               [[VAL_188:%.*]] = addi [[VAL_187]], [[VAL_137]] : index
// CHECK:               [[VAL_189:%.*]] = cmpi "slt", [[VAL_188]], [[VAL_172]] : index
// CHECK:               loop.if [[VAL_189]] {
// CHECK:                 [[VAL_190:%.*]] = load [[VAL_173]]{{\[}}[[VAL_185]], [[VAL_188]]] : memref<?x?xf32, #map3>
// CHECK:                 [[VAL_191:%.*]] = load [[VAL_178]]{{\[}}[[VAL_185]], [[VAL_188]]] : memref<?x?xf32, #map3>
// CHECK:                 [[VAL_192:%.*]] = load [[VAL_183]]{{\[}}[[VAL_185]], [[VAL_188]]] : memref<?x?xf32, #map3>
// CHECK:                 [[VAL_193:%.*]] = addf [[VAL_190]], [[VAL_191]] : f32
// CHECK:                 store [[VAL_193]], [[VAL_183]]{{\[}}[[VAL_185]], [[VAL_188]]] : memref<?x?xf32, #map3>
// CHECK:               }
// CHECK:             }
// CHECK:             gpu.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:       }

