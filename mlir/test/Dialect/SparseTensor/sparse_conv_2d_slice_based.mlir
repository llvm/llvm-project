// RUN: mlir-opt %s --sparsification="enable-index-reduction=true" --canonicalize --cse | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#DCSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>

// CHECK-LABEL:   func.func @conv2d_all_sparse_CSR(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x8xi32, #sparse_tensor.encoding<{{.*}}>>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<3x3xi32>) -> tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant -2 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_11:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_12:.*]] = bufferization.alloc_tensor() : tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<8x8xi32, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<8x8xi32, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_15:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<8x8xi32, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_16:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<8x8xi32, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_17:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x8xi32, #sparse_tensor.encoding<{{.*}}>> to memref<?xi32>
// CHECK-DAG:       %[[VAL_18:.*]] = memref.alloca() : memref<11xindex>
// CHECK-DAG:       %[[VAL_19:.*]] = memref.alloca() : memref<5xindex>
// CHECK-DAG:       %[[VAL_20:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_7]]] : memref<?xindex>
// CHECK:           memref.store %[[VAL_9]], %[[VAL_19]]{{\[}}%[[VAL_6]]] : memref<5xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<5xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_19]]{{\[}}%[[VAL_8]]] : memref<5xindex>
// CHECK:           memref.store %[[VAL_20]], %[[VAL_19]]{{\[}}%[[VAL_5]]] : memref<5xindex>
// CHECK:           %[[VAL_21:.*]] = arith.cmpi ugt, %[[VAL_20]], %[[VAL_6]] : index
// CHECK:           %[[VAL_22:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// CHECK:           %[[VAL_23:.*]] = arith.cmpi uge, %[[VAL_22]], %[[VAL_5]] : index
// CHECK:           %[[VAL_24:.*]] = arith.andi %[[VAL_21]], %[[VAL_23]] : i1
// CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_22]], %[[VAL_3]] : index
// CHECK:           %[[VAL_26:.*]] = arith.select %[[VAL_24]], %[[VAL_25]], %[[VAL_6]] : index
// CHECK:           %[[VAL_27:.*]]:3 = scf.while (%[[VAL_28:.*]] = %[[VAL_21]], %[[VAL_29:.*]] = %[[VAL_22]], %[[VAL_30:.*]] = %[[VAL_26]], %[[VAL_31:.*]] = %[[VAL_12]]) : (i1, index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>) -> (index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>) {
// CHECK:             scf.condition(%[[VAL_28]]) %[[VAL_29]], %[[VAL_30]], %[[VAL_31]] : index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_32:.*]]: index, %[[VAL_33:.*]]: index, %[[VAL_34:.*]]: tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>):
// CHECK:             %[[VAL_35:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_6]]] : memref<5xindex>
// CHECK:             %[[VAL_36:.*]]:4 = scf.for %[[VAL_37:.*]] = %[[VAL_8]] to %[[VAL_35]] step %[[VAL_5]] iter_args(%[[VAL_38:.*]] = %[[VAL_11]], %[[VAL_39:.*]] = %[[VAL_4]], %[[VAL_40:.*]] = %[[VAL_8]], %[[VAL_41:.*]] = %[[VAL_6]]) -> (i1, index, index, index) {
// CHECK:               %[[VAL_42:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_37]]] : memref<5xindex>
// CHECK:               %[[VAL_43:.*]] = arith.addi %[[VAL_37]], %[[VAL_7]] : index
// CHECK:               %[[VAL_44:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_43]]] : memref<5xindex>
// CHECK:               %[[VAL_45:.*]] = arith.addi %[[VAL_37]], %[[VAL_8]] : index
// CHECK:               memref.store %[[VAL_41]], %[[VAL_19]]{{\[}}%[[VAL_45]]] : memref<5xindex>
// CHECK:               %[[VAL_46:.*]] = arith.addi %[[VAL_33]], %[[VAL_5]] : index
// CHECK:               %[[VAL_47:.*]]:5 = scf.while (%[[VAL_48:.*]] = %[[VAL_42]], %[[VAL_49:.*]] = %[[VAL_38]], %[[VAL_50:.*]] = %[[VAL_39]], %[[VAL_51:.*]] = %[[VAL_40]], %[[VAL_52:.*]] = %[[VAL_41]]) : (index, i1, index, index, index) -> (index, i1, index, index, index) {
// CHECK:                 %[[VAL_53:.*]] = arith.cmpi ult, %[[VAL_48]], %[[VAL_44]] : index
// CHECK:                 %[[VAL_54:.*]] = scf.if %[[VAL_53]] -> (i1) {
// CHECK:                   %[[VAL_55:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_48]]] : memref<?xindex>
// CHECK:                   %[[VAL_56:.*]] = arith.cmpi ult, %[[VAL_55]], %[[VAL_46]] : index
// CHECK:                   scf.yield %[[VAL_56]] : i1
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_11]] : i1
// CHECK:                 }
// CHECK:                 scf.condition(%[[VAL_57:.*]]) %[[VAL_48]], %[[VAL_49]], %[[VAL_50]], %[[VAL_51]], %[[VAL_52]] : index, i1, index, index, index
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_58:.*]]: index, %[[VAL_59:.*]]: i1, %[[VAL_60:.*]]: index, %[[VAL_61:.*]]: index, %[[VAL_62:.*]]: index):
// CHECK:                 %[[VAL_63:.*]] = arith.addi %[[VAL_58]], %[[VAL_7]] : index
// CHECK:                 %[[VAL_64:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_58]]] : memref<?xindex>
// CHECK:                 %[[VAL_65:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_63]]] : memref<?xindex>
// CHECK:                 %[[VAL_66:.*]] = arith.cmpi ult, %[[VAL_64]], %[[VAL_65]] : index
// CHECK:                 %[[VAL_67:.*]] = arith.ori %[[VAL_66]], %[[VAL_59]] : i1
// CHECK:                 %[[VAL_68:.*]] = scf.if %[[VAL_66]] -> (index) {
// CHECK:                   %[[VAL_69:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_64]]] : memref<?xindex>
// CHECK:                   %[[VAL_70:.*]] = arith.cmpi ult, %[[VAL_69]], %[[VAL_60]] : index
// CHECK:                   %[[VAL_71:.*]] = arith.select %[[VAL_70]], %[[VAL_69]], %[[VAL_60]] : index
// CHECK:                   scf.yield %[[VAL_71]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_60]] : index
// CHECK:                 }
// CHECK:                 memref.store %[[VAL_64]], %[[VAL_18]]{{\[}}%[[VAL_61]]] : memref<11xindex>
// CHECK:                 %[[VAL_72:.*]] = arith.addi %[[VAL_61]], %[[VAL_7]] : index
// CHECK:                 memref.store %[[VAL_65]], %[[VAL_18]]{{\[}}%[[VAL_72]]] : memref<11xindex>
// CHECK:                 %[[VAL_73:.*]] = arith.addi %[[VAL_61]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_74:.*]] = arith.addi %[[VAL_62]], %[[VAL_7]] : index
// CHECK:                 scf.yield %[[VAL_63]], %[[VAL_67]], %[[VAL_75:.*]], %[[VAL_73]], %[[VAL_74]] : index, i1, index, index, index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_76:.*]]#1, %[[VAL_76]]#2, %[[VAL_76]]#3, %[[VAL_76]]#4 : i1, index, index, index
// CHECK:             }
// CHECK:             memref.store %[[VAL_77:.*]]#2, %[[VAL_18]]{{\[}}%[[VAL_6]]] : memref<11xindex>
// CHECK:             memref.store %[[VAL_6]], %[[VAL_18]]{{\[}}%[[VAL_7]]] : memref<11xindex>
// CHECK:             %[[VAL_78:.*]] = arith.cmpi uge, %[[VAL_77]]#1, %[[VAL_5]] : index
// CHECK:             %[[VAL_79:.*]] = arith.andi %[[VAL_77]]#0, %[[VAL_78]] : i1
// CHECK:             %[[VAL_80:.*]] = arith.addi %[[VAL_77]]#1, %[[VAL_3]] : index
// CHECK:             %[[VAL_81:.*]] = arith.select %[[VAL_79]], %[[VAL_80]], %[[VAL_6]] : index
// CHECK:             %[[VAL_82:.*]]:3 = scf.while (%[[VAL_83:.*]] = %[[VAL_77]]#0, %[[VAL_84:.*]] = %[[VAL_77]]#1, %[[VAL_85:.*]] = %[[VAL_81]], %[[VAL_86:.*]] = %[[VAL_34]]) : (i1, index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>) -> (index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>) {
// CHECK:               scf.condition(%[[VAL_83]]) %[[VAL_84]], %[[VAL_85]], %[[VAL_86]] : index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_87:.*]]: index, %[[VAL_88:.*]]: index, %[[VAL_89:.*]]: tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>):
// CHECK:               %[[VAL_90:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<5xindex>
// CHECK:               %[[VAL_91:.*]] = arith.addi %[[VAL_90]], %[[VAL_8]] : index
// CHECK:               %[[VAL_92:.*]] = arith.addi %[[VAL_90]], %[[VAL_5]] : index
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_91]]] : memref<5xindex>
// CHECK:               %[[VAL_94:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_92]]] : memref<5xindex>
// CHECK:               %[[VAL_95:.*]] = arith.addi %[[VAL_33]], %[[VAL_5]] : index
// CHECK:               %[[VAL_96:.*]]:3 = scf.while (%[[VAL_97:.*]] = %[[VAL_93]], %[[VAL_98:.*]] = %[[VAL_10]], %[[VAL_99:.*]] = %[[VAL_11]]) : (index, i32, i1) -> (index, i32, i1) {
// CHECK:                 %[[VAL_100:.*]] = arith.cmpi ult, %[[VAL_97]], %[[VAL_94]] : index
// CHECK:                 %[[VAL_101:.*]] = scf.if %[[VAL_100]] -> (i1) {
// CHECK:                   %[[VAL_102:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_97]]] : memref<?xindex>
// CHECK:                   %[[VAL_103:.*]] = arith.cmpi ult, %[[VAL_102]], %[[VAL_95]] : index
// CHECK:                   scf.yield %[[VAL_103]] : i1
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_11]] : i1
// CHECK:                 }
// CHECK:                 scf.condition(%[[VAL_104:.*]]) %[[VAL_97]], %[[VAL_98]], %[[VAL_99]] : index, i32, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_105:.*]]: index, %[[VAL_106:.*]]: i32, %[[VAL_107:.*]]: i1):
// CHECK:                 %[[VAL_108:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_105]]] : memref<?xindex>
// CHECK:                 %[[VAL_109:.*]] = arith.subi %[[VAL_108]], %[[VAL_33]] : index
// CHECK:                 %[[VAL_110:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_7]]] : memref<11xindex>
// CHECK:                 %[[VAL_111:.*]] = arith.addi %[[VAL_110]], %[[VAL_8]] : index
// CHECK:                 %[[VAL_112:.*]] = arith.addi %[[VAL_110]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_113:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_111]]] : memref<11xindex>
// CHECK:                 %[[VAL_114:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_112]]] : memref<11xindex>
// CHECK:                 %[[VAL_115:.*]] = arith.addi %[[VAL_88]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_116:.*]]:2 = scf.while (%[[VAL_117:.*]] = %[[VAL_113]], %[[VAL_118:.*]] = %[[VAL_106]]) : (index, i32) -> (index, i32) {
// CHECK:                   %[[VAL_119:.*]] = arith.cmpi ult, %[[VAL_117]], %[[VAL_114]] : index
// CHECK:                   %[[VAL_120:.*]] = scf.if %[[VAL_119]] -> (i1) {
// CHECK:                     %[[VAL_121:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_117]]] : memref<?xindex>
// CHECK:                     %[[VAL_122:.*]] = arith.cmpi ult, %[[VAL_121]], %[[VAL_115]] : index
// CHECK:                     scf.yield %[[VAL_122]] : i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_11]] : i1
// CHECK:                   }
// CHECK:                   scf.condition(%[[VAL_123:.*]]) %[[VAL_117]], %[[VAL_118]] : index, i32
// CHECK:                 } do {
// CHECK:                 ^bb0(%[[VAL_124:.*]]: index, %[[VAL_125:.*]]: i32):
// CHECK:                   %[[VAL_126:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_124]]] : memref<?xindex>
// CHECK:                   %[[VAL_127:.*]] = arith.subi %[[VAL_126]], %[[VAL_88]] : index
// CHECK:                   %[[VAL_128:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_124]]] : memref<?xi32>
// CHECK:                   %[[VAL_129:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_109]], %[[VAL_127]]] : tensor<3x3xi32>
// CHECK:                   %[[VAL_130:.*]] = arith.muli %[[VAL_128]], %[[VAL_129]] : i32
// CHECK:                   %[[VAL_131:.*]] = arith.addi %[[VAL_125]], %[[VAL_130]] : i32
// CHECK:                   %[[VAL_132:.*]] = arith.addi %[[VAL_124]], %[[VAL_7]] : index
// CHECK:                   scf.yield %[[VAL_132]], %[[VAL_131]] : index, i32
// CHECK:                 }
// CHECK:                 %[[VAL_133:.*]] = arith.addi %[[VAL_105]], %[[VAL_7]] : index
// CHECK:                 memref.store %[[VAL_112]], %[[VAL_18]]{{\[}}%[[VAL_7]]] : memref<11xindex>
// CHECK:                 scf.yield %[[VAL_133]], %[[VAL_136:.*]]#1, %[[VAL_2]] : index, i32, i1
// CHECK:               }
// CHECK:               %[[VAL_137:.*]] = scf.if %[[VAL_138:.*]]#2 -> (tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>) {
// CHECK:                 %[[VAL_139:.*]] = sparse_tensor.insert %[[VAL_138]]#1 into %[[VAL_89]]{{\[}}%[[VAL_33]], %[[VAL_88]]] : tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:                 scf.yield %[[VAL_139]] : tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_89]] : tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:               }
// CHECK:               memref.store %[[VAL_6]], %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<5xindex>
// CHECK:               memref.store %[[VAL_6]], %[[VAL_18]]{{\[}}%[[VAL_7]]] : memref<11xindex>
// CHECK:               %[[VAL_140:.*]] = arith.cmpi ugt, %[[VAL_87]], %[[VAL_88]] : index
// CHECK:               %[[VAL_141:.*]]:3 = scf.if %[[VAL_140]] -> (index, i1, index) {
// CHECK:                 %[[VAL_142:.*]] = arith.addi %[[VAL_88]], %[[VAL_7]] : index
// CHECK:                 scf.yield %[[VAL_87]], %[[VAL_2]], %[[VAL_142]] : index, i1, index
// CHECK:               } else {
// CHECK:                 %[[VAL_143:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_6]]] : memref<11xindex>
// CHECK:                 %[[VAL_144:.*]]:2 = scf.for %[[VAL_145:.*]] = %[[VAL_8]] to %[[VAL_143]] step %[[VAL_5]] iter_args(%[[VAL_146:.*]] = %[[VAL_4]], %[[VAL_147:.*]] = %[[VAL_11]]) -> (index, i1) {
// CHECK:                   %[[VAL_148:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_145]]] : memref<11xindex>
// CHECK:                   %[[VAL_149:.*]] = arith.addi %[[VAL_145]], %[[VAL_7]] : index
// CHECK:                   %[[VAL_150:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_149]]] : memref<11xindex>
// CHECK:                   %[[VAL_151:.*]] = arith.cmpi ult, %[[VAL_148]], %[[VAL_150]] : index
// CHECK:                   %[[VAL_152:.*]] = scf.if %[[VAL_151]] -> (index) {
// CHECK:                     %[[VAL_153:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_148]]] : memref<?xindex>
// CHECK:                     %[[VAL_154:.*]] = arith.cmpi eq, %[[VAL_153]], %[[VAL_87]] : index
// CHECK:                     %[[VAL_155:.*]] = scf.if %[[VAL_154]] -> (index) {
// CHECK:                       %[[VAL_156:.*]] = arith.addi %[[VAL_148]], %[[VAL_7]] : index
// CHECK:                       memref.store %[[VAL_156]], %[[VAL_18]]{{\[}}%[[VAL_145]]] : memref<11xindex>
// CHECK:                       scf.yield %[[VAL_156]] : index
// CHECK:                     } else {
// CHECK:                       scf.yield %[[VAL_148]] : index
// CHECK:                     }
// CHECK:                     scf.yield %[[VAL_157:.*]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_148]] : index
// CHECK:                   }
// CHECK:                   %[[VAL_158:.*]] = arith.cmpi ult, %[[VAL_159:.*]], %[[VAL_150]] : index
// CHECK:                   %[[VAL_160:.*]] = scf.if %[[VAL_158]] -> (index) {
// CHECK:                     %[[VAL_161:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_159]]] : memref<?xindex>
// CHECK:                     scf.yield %[[VAL_161]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_146]] : index
// CHECK:                   }
// CHECK:                   %[[VAL_162:.*]] = arith.ori %[[VAL_158]], %[[VAL_147]] : i1
// CHECK:                   %[[VAL_163:.*]] = arith.cmpi ult, %[[VAL_164:.*]], %[[VAL_146]] : index
// CHECK:                   %[[VAL_165:.*]] = arith.select %[[VAL_163]], %[[VAL_164]], %[[VAL_146]] : index
// CHECK:                   scf.yield %[[VAL_165]], %[[VAL_162]] : index, i1
// CHECK:                 }
// CHECK:                 %[[VAL_166:.*]] = arith.addi %[[VAL_167:.*]]#0, %[[VAL_7]] : index
// CHECK:                 %[[VAL_168:.*]] = arith.addi %[[VAL_167]]#0, %[[VAL_3]] : index
// CHECK:                 %[[VAL_169:.*]] = arith.cmpi uge, %[[VAL_166]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_170:.*]] = arith.select %[[VAL_169]], %[[VAL_168]], %[[VAL_6]] : index
// CHECK:                 scf.yield %[[VAL_167]]#0, %[[VAL_167]]#1, %[[VAL_170]] : index, i1, index
// CHECK:               }
// CHECK:               %[[VAL_171:.*]] = arith.addi %[[VAL_88]], %[[VAL_7]] : index
// CHECK:               %[[VAL_172:.*]] = arith.cmpi ugt, %[[VAL_173:.*]]#2, %[[VAL_171]] : index
// CHECK:               %[[VAL_174:.*]] = arith.select %[[VAL_172]], %[[VAL_173]]#2, %[[VAL_171]] : index
// CHECK:               %[[VAL_175:.*]] = arith.addi %[[VAL_174]], %[[VAL_5]] : index
// CHECK:               %[[VAL_176:.*]] = arith.cmpi ule, %[[VAL_175]], %[[VAL_4]] : index
// CHECK:               %[[VAL_177:.*]] = arith.andi %[[VAL_173]]#1, %[[VAL_176]] : i1
// CHECK:               scf.yield %[[VAL_177]], %[[VAL_173]]#0, %[[VAL_174]], %[[VAL_178:.*]] : i1, index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:             }
// CHECK:             memref.store %[[VAL_6]], %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<5xindex>
// CHECK:             %[[VAL_179:.*]] = arith.cmpi ugt, %[[VAL_32]], %[[VAL_33]] : index
// CHECK:             %[[VAL_180:.*]]:3 = scf.if %[[VAL_179]] -> (index, i1, index) {
// CHECK:               %[[VAL_181:.*]] = arith.addi %[[VAL_33]], %[[VAL_7]] : index
// CHECK:               scf.yield %[[VAL_32]], %[[VAL_2]], %[[VAL_181]] : index, i1, index
// CHECK:             } else {
// CHECK:               %[[VAL_182:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_6]]] : memref<5xindex>
// CHECK:               %[[VAL_183:.*]]:2 = scf.for %[[VAL_184:.*]] = %[[VAL_8]] to %[[VAL_182]] step %[[VAL_5]] iter_args(%[[VAL_185:.*]] = %[[VAL_4]], %[[VAL_186:.*]] = %[[VAL_11]]) -> (index, i1) {
// CHECK:                 %[[VAL_187:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_184]]] : memref<5xindex>
// CHECK:                 %[[VAL_188:.*]] = arith.addi %[[VAL_184]], %[[VAL_7]] : index
// CHECK:                 %[[VAL_189:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_188]]] : memref<5xindex>
// CHECK:                 %[[VAL_190:.*]] = arith.cmpi ult, %[[VAL_187]], %[[VAL_189]] : index
// CHECK:                 %[[VAL_191:.*]] = scf.if %[[VAL_190]] -> (index) {
// CHECK:                   %[[VAL_192:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_187]]] : memref<?xindex>
// CHECK:                   %[[VAL_193:.*]] = arith.cmpi eq, %[[VAL_192]], %[[VAL_32]] : index
// CHECK:                   %[[VAL_194:.*]] = scf.if %[[VAL_193]] -> (index) {
// CHECK:                     %[[VAL_195:.*]] = arith.addi %[[VAL_187]], %[[VAL_7]] : index
// CHECK:                     memref.store %[[VAL_195]], %[[VAL_19]]{{\[}}%[[VAL_184]]] : memref<5xindex>
// CHECK:                     scf.yield %[[VAL_195]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_187]] : index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_196:.*]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_187]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_197:.*]] = arith.cmpi ult, %[[VAL_198:.*]], %[[VAL_189]] : index
// CHECK:                 %[[VAL_199:.*]] = scf.if %[[VAL_197]] -> (index) {
// CHECK:                   %[[VAL_200:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_198]]] : memref<?xindex>
// CHECK:                   scf.yield %[[VAL_200]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_185]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_201:.*]] = arith.ori %[[VAL_197]], %[[VAL_186]] : i1
// CHECK:                 %[[VAL_202:.*]] = arith.cmpi ult, %[[VAL_203:.*]], %[[VAL_185]] : index
// CHECK:                 %[[VAL_204:.*]] = arith.select %[[VAL_202]], %[[VAL_203]], %[[VAL_185]] : index
// CHECK:                 scf.yield %[[VAL_204]], %[[VAL_201]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_205:.*]] = arith.addi %[[VAL_206:.*]]#0, %[[VAL_7]] : index
// CHECK:               %[[VAL_207:.*]] = arith.addi %[[VAL_206]]#0, %[[VAL_3]] : index
// CHECK:               %[[VAL_208:.*]] = arith.cmpi uge, %[[VAL_205]], %[[VAL_5]] : index
// CHECK:               %[[VAL_209:.*]] = arith.select %[[VAL_208]], %[[VAL_207]], %[[VAL_6]] : index
// CHECK:               scf.yield %[[VAL_206]]#0, %[[VAL_206]]#1, %[[VAL_209]] : index, i1, index
// CHECK:             }
// CHECK:             %[[VAL_210:.*]] = arith.addi %[[VAL_33]], %[[VAL_7]] : index
// CHECK:             %[[VAL_211:.*]] = arith.cmpi ugt, %[[VAL_212:.*]]#2, %[[VAL_210]] : index
// CHECK:             %[[VAL_213:.*]] = arith.select %[[VAL_211]], %[[VAL_212]]#2, %[[VAL_210]] : index
// CHECK:             %[[VAL_214:.*]] = arith.addi %[[VAL_213]], %[[VAL_5]] : index
// CHECK:             %[[VAL_215:.*]] = arith.cmpi ule, %[[VAL_214]], %[[VAL_4]] : index
// CHECK:             %[[VAL_216:.*]] = arith.andi %[[VAL_212]]#1, %[[VAL_215]] : i1
// CHECK:             scf.yield %[[VAL_216]], %[[VAL_212]]#0, %[[VAL_213]], %[[VAL_217:.*]]#2 : i1, index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           }
// CHECK:           %[[VAL_218:.*]] = sparse_tensor.load %[[VAL_219:.*]]#2 hasInserts : tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           return %[[VAL_218]] : tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:         }
func.func @conv2d_all_sparse_CSR(%arg0: tensor<8x8xi32, #DCSR>,
                                 %arg1: tensor<3x3xi32>) -> tensor<6x6xi32, #DCSR> {
  %0 = bufferization.alloc_tensor() : tensor<6x6xi32, #DCSR>
  %1 = linalg.generic {
         indexing_maps = [#map, #map1, #map2],
         iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
         ins(%arg0, %arg1 : tensor<8x8xi32, #DCSR>, tensor<3x3xi32>)
         outs(%0 : tensor<6x6xi32, #DCSR>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %2 = arith.muli %in, %in_0 : i32
      %3 = arith.addi %out, %2 : i32
      linalg.yield %3 : i32
    } -> tensor<6x6xi32, #DCSR>
  return %1 : tensor<6x6xi32, #DCSR>
}
