// RUN: mlir-opt %s --sparsification="enable-index-reduction=true" --canonicalize --cse | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#DCSR = #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>
// CHECK-LABEL:   func.func @conv2d_all_sparse_CSR(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x8xi32, #sparse_tensor.encoding<{{.*}}>>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<3x3xi32>)
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant -2 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_11:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_12:.*]] = bufferization.alloc_tensor() : tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<8x8xi32, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<8x8xi32, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_15:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<8x8xi32, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_16:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<8x8xi32, #sparse_tensor.encoding<{{.*}}>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_17:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x8xi32, #sparse_tensor.encoding<{{.*}}>> to memref<?xi32>
// CHECK-DAG:       %[[VAL_18:.*]] = memref.alloca() : memref<8xindex>
// CHECK-DAG:       %[[VAL_19:.*]] = memref.alloca() : memref<4xindex>
// CHECK-DAG:       %[[VAL_20:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_9]]] : memref<?xindex>
// CHECK:           memref.store %[[VAL_6]], %[[VAL_19]]{{\[}}%[[VAL_8]]] : memref<4xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_19]]{{\[}}%[[VAL_9]]] : memref<4xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<4xindex>
// CHECK:           memref.store %[[VAL_20]], %[[VAL_19]]{{\[}}%[[VAL_5]]] : memref<4xindex>
// CHECK:           %[[VAL_21:.*]] = arith.cmpi ugt, %[[VAL_20]], %[[VAL_8]] : index
// CHECK:           %[[VAL_22:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_8]]] : memref<?xindex>
// CHECK:           %[[VAL_23:.*]] = arith.cmpi uge, %[[VAL_22]], %[[VAL_5]] : index
// CHECK:           %[[VAL_24:.*]] = arith.andi %[[VAL_21]], %[[VAL_23]] : i1
// CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_22]], %[[VAL_3]] : index
// CHECK:           %[[VAL_26:.*]] = arith.select %[[VAL_24]], %[[VAL_25]], %[[VAL_8]] : index
// CHECK:           %[[VAL_27:.*]]:3 = scf.while (%[[VAL_28:.*]] = %[[VAL_21]], %[[VAL_29:.*]] = %[[VAL_22]], %[[VAL_30:.*]] = %[[VAL_26]], %[[VAL_31:.*]] = %[[VAL_12]]) : (i1, index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>) -> (index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>) {
// CHECK:             scf.condition(%[[VAL_28]]) %[[VAL_29]], %[[VAL_30]], %[[VAL_31]] : index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_32:.*]]: index, %[[VAL_33:.*]]: index, %[[VAL_34:.*]]: tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>):
// CHECK:             %[[VAL_35:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_8]]] : memref<4xindex>
// CHECK:             %[[VAL_36:.*]]:3 = scf.for %[[VAL_37:.*]] = %[[VAL_7]] to %[[VAL_35]] step %[[VAL_7]] iter_args(%[[VAL_38:.*]] = %[[VAL_11]], %[[VAL_39:.*]] = %[[VAL_4]], %[[VAL_40:.*]] = %[[VAL_7]]) -> (i1, index, index) {
// CHECK:               %[[VAL_41:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_37]]] : memref<4xindex>
// CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_37]], %[[VAL_9]] : index
// CHECK:               %[[VAL_43:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_42]]] : memref<4xindex>
// CHECK:               %[[VAL_44:.*]] = arith.addi %[[VAL_33]], %[[VAL_5]] : index
// CHECK:               %[[VAL_45:.*]]:4 = scf.while (%[[VAL_46:.*]] = %[[VAL_41]], %[[VAL_47:.*]] = %[[VAL_38]], %[[VAL_48:.*]] = %[[VAL_39]], %[[VAL_49:.*]] = %[[VAL_40]]) : (index, i1, index, index) -> (index, i1, index, index) {
// CHECK:                 %[[VAL_50:.*]] = arith.cmpi ult, %[[VAL_46]], %[[VAL_43]] : index
// CHECK:                 %[[VAL_51:.*]] = scf.if %[[VAL_50]] -> (i1) {
// CHECK:                   %[[VAL_52:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_46]]] : memref<?xindex>
// CHECK:                   %[[VAL_53:.*]] = arith.cmpi ult, %[[VAL_52]], %[[VAL_44]] : index
// CHECK:                   scf.yield %[[VAL_53]] : i1
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_11]] : i1
// CHECK:                 }
// CHECK:                 scf.condition(%[[VAL_54:.*]]) %[[VAL_46]], %[[VAL_47]], %[[VAL_48]], %[[VAL_49]] : index, i1, index, index
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_55:.*]]: index, %[[VAL_56:.*]]: i1, %[[VAL_57:.*]]: index, %[[VAL_58:.*]]: index):
// CHECK:                 %[[VAL_59:.*]] = arith.addi %[[VAL_55]], %[[VAL_9]] : index
// CHECK:                 %[[VAL_60:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_55]]] : memref<?xindex>
// CHECK:                 %[[VAL_61:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_59]]] : memref<?xindex>
// CHECK:                 %[[VAL_62:.*]] = arith.cmpi ult, %[[VAL_60]], %[[VAL_61]] : index
// CHECK:                 %[[VAL_63:.*]] = arith.ori %[[VAL_62]], %[[VAL_56]] : i1
// CHECK:                 %[[VAL_64:.*]] = scf.if %[[VAL_62]] -> (index) {
// CHECK:                   %[[VAL_65:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_60]]] : memref<?xindex>
// CHECK:                   %[[VAL_66:.*]] = arith.cmpi ult, %[[VAL_65]], %[[VAL_57]] : index
// CHECK:                   %[[VAL_67:.*]] = arith.select %[[VAL_66]], %[[VAL_65]], %[[VAL_57]] : index
// CHECK:                   scf.yield %[[VAL_67]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_57]] : index
// CHECK:                 }
// CHECK:                 memref.store %[[VAL_60]], %[[VAL_18]]{{\[}}%[[VAL_58]]] : memref<8xindex>
// CHECK:                 %[[VAL_68:.*]] = arith.addi %[[VAL_58]], %[[VAL_9]] : index
// CHECK:                 memref.store %[[VAL_61]], %[[VAL_18]]{{\[}}%[[VAL_68]]] : memref<8xindex>
// CHECK:                 %[[VAL_69:.*]] = arith.addi %[[VAL_58]], %[[VAL_7]] : index
// CHECK:                 scf.yield %[[VAL_59]], %[[VAL_63]], %[[VAL_70:.*]], %[[VAL_69]] : index, i1, index, index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_71:.*]]#1, %[[VAL_71]]#2, %[[VAL_71]]#3 : i1, index, index
// CHECK:             }
// CHECK:             memref.store %[[VAL_72:.*]]#2, %[[VAL_18]]{{\[}}%[[VAL_8]]] : memref<8xindex>
// CHECK:             memref.store %[[VAL_8]], %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<8xindex>
// CHECK:             %[[VAL_73:.*]] = arith.cmpi uge, %[[VAL_72]]#1, %[[VAL_5]] : index
// CHECK:             %[[VAL_74:.*]] = arith.andi %[[VAL_72]]#0, %[[VAL_73]] : i1
// CHECK:             %[[VAL_75:.*]] = arith.addi %[[VAL_72]]#1, %[[VAL_3]] : index
// CHECK:             %[[VAL_76:.*]] = arith.select %[[VAL_74]], %[[VAL_75]], %[[VAL_8]] : index
// CHECK:             %[[VAL_77:.*]]:3 = scf.while (%[[VAL_78:.*]] = %[[VAL_72]]#0, %[[VAL_79:.*]] = %[[VAL_72]]#1, %[[VAL_80:.*]] = %[[VAL_76]], %[[VAL_81:.*]] = %[[VAL_34]]) : (i1, index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>) -> (index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>) {
// CHECK:               scf.condition(%[[VAL_78]]) %[[VAL_79]], %[[VAL_80]], %[[VAL_81]] : index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_82:.*]]: index, %[[VAL_83:.*]]: index, %[[VAL_84:.*]]: tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>):
// CHECK:               %[[VAL_85:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_9]]] : memref<4xindex>
// CHECK:               %[[VAL_86:.*]] = arith.addi %[[VAL_85]], %[[VAL_7]] : index
// CHECK:               %[[VAL_87:.*]] = arith.addi %[[VAL_85]], %[[VAL_5]] : index
// CHECK:               %[[VAL_88:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_86]]] : memref<4xindex>
// CHECK:               %[[VAL_89:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_87]]] : memref<4xindex>
// CHECK:               %[[VAL_90:.*]] = arith.addi %[[VAL_33]], %[[VAL_5]] : index
// CHECK:               %[[VAL_91:.*]]:3 = scf.while (%[[VAL_92:.*]] = %[[VAL_88]], %[[VAL_93:.*]] = %[[VAL_10]], %[[VAL_94:.*]] = %[[VAL_11]]) : (index, i32, i1) -> (index, i32, i1) {
// CHECK:                 %[[VAL_95:.*]] = arith.cmpi ult, %[[VAL_92]], %[[VAL_89]] : index
// CHECK:                 %[[VAL_96:.*]] = scf.if %[[VAL_95]] -> (i1) {
// CHECK:                   %[[VAL_97:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_92]]] : memref<?xindex>
// CHECK:                   %[[VAL_98:.*]] = arith.cmpi ult, %[[VAL_97]], %[[VAL_90]] : index
// CHECK:                   scf.yield %[[VAL_98]] : i1
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_11]] : i1
// CHECK:                 }
// CHECK:                 scf.condition(%[[VAL_99:.*]]) %[[VAL_92]], %[[VAL_93]], %[[VAL_94]] : index, i32, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_100:.*]]: index, %[[VAL_101:.*]]: i32, %[[VAL_102:.*]]: i1):
// CHECK:                 %[[VAL_103:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_100]]] : memref<?xindex>
// CHECK:                 %[[VAL_104:.*]] = arith.subi %[[VAL_103]], %[[VAL_33]] : index
// CHECK:                 %[[VAL_105:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<8xindex>
// CHECK:                 %[[VAL_106:.*]] = arith.addi %[[VAL_105]], %[[VAL_7]] : index
// CHECK:                 %[[VAL_107:.*]] = arith.addi %[[VAL_105]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_108:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_106]]] : memref<8xindex>
// CHECK:                 %[[VAL_109:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_107]]] : memref<8xindex>
// CHECK:                 %[[VAL_110:.*]] = arith.addi %[[VAL_83]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_111:.*]]:2 = scf.while (%[[VAL_112:.*]] = %[[VAL_108]], %[[VAL_113:.*]] = %[[VAL_101]]) : (index, i32) -> (index, i32) {
// CHECK:                   %[[VAL_114:.*]] = arith.cmpi ult, %[[VAL_112]], %[[VAL_109]] : index
// CHECK:                   %[[VAL_115:.*]] = scf.if %[[VAL_114]] -> (i1) {
// CHECK:                     %[[VAL_116:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_112]]] : memref<?xindex>
// CHECK:                     %[[VAL_117:.*]] = arith.cmpi ult, %[[VAL_116]], %[[VAL_110]] : index
// CHECK:                     scf.yield %[[VAL_117]] : i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_11]] : i1
// CHECK:                   }
// CHECK:                   scf.condition(%[[VAL_118:.*]]) %[[VAL_112]], %[[VAL_113]] : index, i32
// CHECK:                 } do {
// CHECK:                 ^bb0(%[[VAL_119:.*]]: index, %[[VAL_120:.*]]: i32):
// CHECK:                   %[[VAL_121:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_119]]] : memref<?xindex>
// CHECK:                   %[[VAL_122:.*]] = arith.subi %[[VAL_121]], %[[VAL_83]] : index
// CHECK:                   %[[VAL_123:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_119]]] : memref<?xi32>
// CHECK:                   %[[VAL_124:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_104]], %[[VAL_122]]] : tensor<3x3xi32>
// CHECK:                   %[[VAL_125:.*]] = arith.muli %[[VAL_123]], %[[VAL_124]] : i32
// CHECK:                   %[[VAL_126:.*]] = arith.addi %[[VAL_120]], %[[VAL_125]] : i32
// CHECK:                   %[[VAL_127:.*]] = arith.addi %[[VAL_119]], %[[VAL_9]] : index
// CHECK:                   scf.yield %[[VAL_127]], %[[VAL_126]] : index, i32
// CHECK:                 }
// CHECK:                 %[[VAL_128:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<8xindex>
// CHECK:                 %[[VAL_129:.*]] = arith.addi %[[VAL_128]], %[[VAL_7]] : index
// CHECK:                 memref.store %[[VAL_129]], %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<8xindex>
// CHECK:                 %[[VAL_130:.*]] = arith.addi %[[VAL_100]], %[[VAL_9]] : index
// CHECK:                 scf.yield %[[VAL_130]], %[[VAL_131:.*]]#1, %[[VAL_2]] : index, i32, i1
// CHECK:               }
// CHECK:               %[[VAL_132:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_9]]] : memref<4xindex>
// CHECK:               %[[VAL_133:.*]] = arith.addi %[[VAL_132]], %[[VAL_7]] : index
// CHECK:               memref.store %[[VAL_133]], %[[VAL_19]]{{\[}}%[[VAL_9]]] : memref<4xindex>
// CHECK:               %[[VAL_134:.*]] = scf.if %[[VAL_135:.*]]#2 -> (tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>) {
// CHECK:                 %[[VAL_136:.*]] = sparse_tensor.insert %[[VAL_135]]#1 into %[[VAL_84]]{{\[}}%[[VAL_33]], %[[VAL_83]]] : tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:                 scf.yield %[[VAL_136]] : tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_84]] : tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:               }
// CHECK:               memref.store %[[VAL_8]], %[[VAL_19]]{{\[}}%[[VAL_9]]] : memref<4xindex>
// CHECK:               memref.store %[[VAL_8]], %[[VAL_18]]{{\[}}%[[VAL_9]]] : memref<8xindex>
// CHECK:               %[[VAL_137:.*]] = arith.cmpi ugt, %[[VAL_82]], %[[VAL_83]] : index
// CHECK:               %[[VAL_138:.*]]:3 = scf.if %[[VAL_137]] -> (index, i1, index) {
// CHECK:                 %[[VAL_139:.*]] = arith.addi %[[VAL_83]], %[[VAL_9]] : index
// CHECK:                 scf.yield %[[VAL_82]], %[[VAL_2]], %[[VAL_139]] : index, i1, index
// CHECK:               } else {
// CHECK:                 %[[VAL_140:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_8]]] : memref<8xindex>
// CHECK:                 %[[VAL_141:.*]]:2 = scf.for %[[VAL_142:.*]] = %[[VAL_7]] to %[[VAL_140]] step %[[VAL_7]] iter_args(%[[VAL_143:.*]] = %[[VAL_4]], %[[VAL_144:.*]] = %[[VAL_11]]) -> (index, i1) {
// CHECK:                   %[[VAL_145:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_142]]] : memref<8xindex>
// CHECK:                   %[[VAL_146:.*]] = arith.addi %[[VAL_142]], %[[VAL_9]] : index
// CHECK:                   %[[VAL_147:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_146]]] : memref<8xindex>
// CHECK:                   %[[VAL_148:.*]] = arith.cmpi ult, %[[VAL_145]], %[[VAL_147]] : index
// CHECK:                   %[[VAL_149:.*]] = scf.if %[[VAL_148]] -> (index) {
// CHECK:                     %[[VAL_150:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_145]]] : memref<?xindex>
// CHECK:                     %[[VAL_151:.*]] = arith.cmpi eq, %[[VAL_150]], %[[VAL_82]] : index
// CHECK:                     %[[VAL_152:.*]] = scf.if %[[VAL_151]] -> (index) {
// CHECK:                       %[[VAL_153:.*]] = arith.addi %[[VAL_145]], %[[VAL_9]] : index
// CHECK:                       memref.store %[[VAL_153]], %[[VAL_18]]{{\[}}%[[VAL_142]]] : memref<8xindex>
// CHECK:                       scf.yield %[[VAL_153]] : index
// CHECK:                     } else {
// CHECK:                       scf.yield %[[VAL_145]] : index
// CHECK:                     }
// CHECK:                     scf.yield %[[VAL_154:.*]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_145]] : index
// CHECK:                   }
// CHECK:                   %[[VAL_155:.*]] = arith.cmpi ult, %[[VAL_156:.*]], %[[VAL_147]] : index
// CHECK:                   %[[VAL_157:.*]] = scf.if %[[VAL_155]] -> (index) {
// CHECK:                     %[[VAL_158:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_156]]] : memref<?xindex>
// CHECK:                     scf.yield %[[VAL_158]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_143]] : index
// CHECK:                   }
// CHECK:                   %[[VAL_159:.*]] = arith.ori %[[VAL_155]], %[[VAL_144]] : i1
// CHECK:                   %[[VAL_160:.*]] = arith.cmpi ult, %[[VAL_161:.*]], %[[VAL_143]] : index
// CHECK:                   %[[VAL_162:.*]] = arith.select %[[VAL_160]], %[[VAL_161]], %[[VAL_143]] : index
// CHECK:                   scf.yield %[[VAL_162]], %[[VAL_159]] : index, i1
// CHECK:                 }
// CHECK:                 %[[VAL_163:.*]] = arith.addi %[[VAL_164:.*]]#0, %[[VAL_9]] : index
// CHECK:                 %[[VAL_165:.*]] = arith.addi %[[VAL_164]]#0, %[[VAL_3]] : index
// CHECK:                 %[[VAL_166:.*]] = arith.cmpi uge, %[[VAL_163]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_167:.*]] = arith.select %[[VAL_166]], %[[VAL_165]], %[[VAL_8]] : index
// CHECK:                 scf.yield %[[VAL_164]]#0, %[[VAL_164]]#1, %[[VAL_167]] : index, i1, index
// CHECK:               }
// CHECK:               %[[VAL_168:.*]] = arith.addi %[[VAL_83]], %[[VAL_9]] : index
// CHECK:               %[[VAL_169:.*]] = arith.cmpi ugt, %[[VAL_170:.*]]#2, %[[VAL_168]] : index
// CHECK:               %[[VAL_171:.*]] = arith.select %[[VAL_169]], %[[VAL_170]]#2, %[[VAL_168]] : index
// CHECK:               %[[VAL_172:.*]] = arith.addi %[[VAL_171]], %[[VAL_5]] : index
// CHECK:               %[[VAL_173:.*]] = arith.cmpi ule, %[[VAL_172]], %[[VAL_4]] : index
// CHECK:               %[[VAL_174:.*]] = arith.andi %[[VAL_170]]#1, %[[VAL_173]] : i1
// CHECK:               scf.yield %[[VAL_174]], %[[VAL_170]]#0, %[[VAL_171]], %[[VAL_175:.*]] : i1, index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:             }
// CHECK:             memref.store %[[VAL_8]], %[[VAL_19]]{{\[}}%[[VAL_9]]] : memref<4xindex>
// CHECK:             %[[VAL_176:.*]] = arith.cmpi ugt, %[[VAL_32]], %[[VAL_33]] : index
// CHECK:             %[[VAL_177:.*]]:3 = scf.if %[[VAL_176]] -> (index, i1, index) {
// CHECK:               %[[VAL_178:.*]] = arith.addi %[[VAL_33]], %[[VAL_9]] : index
// CHECK:               scf.yield %[[VAL_32]], %[[VAL_2]], %[[VAL_178]] : index, i1, index
// CHECK:             } else {
// CHECK:               %[[VAL_179:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_8]]] : memref<4xindex>
// CHECK:               %[[VAL_180:.*]]:2 = scf.for %[[VAL_181:.*]] = %[[VAL_7]] to %[[VAL_179]] step %[[VAL_7]] iter_args(%[[VAL_182:.*]] = %[[VAL_4]], %[[VAL_183:.*]] = %[[VAL_11]]) -> (index, i1) {
// CHECK:                 %[[VAL_184:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_181]]] : memref<4xindex>
// CHECK:                 %[[VAL_185:.*]] = arith.addi %[[VAL_181]], %[[VAL_9]] : index
// CHECK:                 %[[VAL_186:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_185]]] : memref<4xindex>
// CHECK:                 %[[VAL_187:.*]] = arith.cmpi ult, %[[VAL_184]], %[[VAL_186]] : index
// CHECK:                 %[[VAL_188:.*]] = scf.if %[[VAL_187]] -> (index) {
// CHECK:                   %[[VAL_189:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_184]]] : memref<?xindex>
// CHECK:                   %[[VAL_190:.*]] = arith.cmpi eq, %[[VAL_189]], %[[VAL_32]] : index
// CHECK:                   %[[VAL_191:.*]] = scf.if %[[VAL_190]] -> (index) {
// CHECK:                     %[[VAL_192:.*]] = arith.addi %[[VAL_184]], %[[VAL_9]] : index
// CHECK:                     memref.store %[[VAL_192]], %[[VAL_19]]{{\[}}%[[VAL_181]]] : memref<4xindex>
// CHECK:                     scf.yield %[[VAL_192]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_184]] : index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_193:.*]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_184]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_194:.*]] = arith.cmpi ult, %[[VAL_195:.*]], %[[VAL_186]] : index
// CHECK:                 %[[VAL_196:.*]] = scf.if %[[VAL_194]] -> (index) {
// CHECK:                   %[[VAL_197:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_195]]] : memref<?xindex>
// CHECK:                   scf.yield %[[VAL_197]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_182]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_198:.*]] = arith.ori %[[VAL_194]], %[[VAL_183]] : i1
// CHECK:                 %[[VAL_199:.*]] = arith.cmpi ult, %[[VAL_200:.*]], %[[VAL_182]] : index
// CHECK:                 %[[VAL_201:.*]] = arith.select %[[VAL_199]], %[[VAL_200]], %[[VAL_182]] : index
// CHECK:                 scf.yield %[[VAL_201]], %[[VAL_198]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_202:.*]] = arith.addi %[[VAL_203:.*]]#0, %[[VAL_9]] : index
// CHECK:               %[[VAL_204:.*]] = arith.addi %[[VAL_203]]#0, %[[VAL_3]] : index
// CHECK:               %[[VAL_205:.*]] = arith.cmpi uge, %[[VAL_202]], %[[VAL_5]] : index
// CHECK:               %[[VAL_206:.*]] = arith.select %[[VAL_205]], %[[VAL_204]], %[[VAL_8]] : index
// CHECK:               scf.yield %[[VAL_203]]#0, %[[VAL_203]]#1, %[[VAL_206]] : index, i1, index
// CHECK:             }
// CHECK:             %[[VAL_207:.*]] = arith.addi %[[VAL_33]], %[[VAL_9]] : index
// CHECK:             %[[VAL_208:.*]] = arith.cmpi ugt, %[[VAL_209:.*]]#2, %[[VAL_207]] : index
// CHECK:             %[[VAL_210:.*]] = arith.select %[[VAL_208]], %[[VAL_209]]#2, %[[VAL_207]] : index
// CHECK:             %[[VAL_211:.*]] = arith.addi %[[VAL_210]], %[[VAL_5]] : index
// CHECK:             %[[VAL_212:.*]] = arith.cmpi ule, %[[VAL_211]], %[[VAL_4]] : index
// CHECK:             %[[VAL_213:.*]] = arith.andi %[[VAL_209]]#1, %[[VAL_212]] : i1
// CHECK:             scf.yield %[[VAL_213]], %[[VAL_209]]#0, %[[VAL_210]], %[[VAL_214:.*]]#2 : i1, index, index, tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           }
// CHECK:           %[[VAL_215:.*]] = sparse_tensor.load %[[VAL_216:.*]]#2 hasInserts : tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
// CHECK:           return %[[VAL_215]] : tensor<6x6xi32, #sparse_tensor.encoding<{{.*}}>>
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
