// RUN: mlir-opt %s --sparse-reinterpret-map --sparsification --canonicalize --cse | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#DCSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>


// CHECK-LABEL:   func.func @conv2d_all_sparse_CSR(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x8xi32, #sparse>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<3x3xi32>) -> tensor<6x6xi32, #sparse> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant -2 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_11:.*]] = tensor.empty() : tensor<6x6xi32, #sparse>
// CHECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<8x8xi32, #sparse> to memref<?xindex>
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<8x8xi32, #sparse> to memref<?xindex>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<8x8xi32, #sparse> to memref<?xindex>
// CHECK-DAG:       %[[VAL_15:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<8x8xi32, #sparse> to memref<?xindex>
// CHECK-DAG:       %[[VAL_16:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x8xi32, #sparse> to memref<?xi32>
// CHECK-DAG:       %[[VAL_17:.*]] = memref.alloca() : memref<9xindex>
// CHECK-DAG:       %[[VAL_18:.*]] = memref.alloca() : memref<3xindex>
// CHECK-DAG:       %[[VAL_19:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_7]]] : memref<?xindex>
// CHECK:           memref.store %[[VAL_8]], %[[VAL_18]]{{\[}}%[[VAL_8]]] : memref<3xindex>
// CHECK:           memref.store %[[VAL_19]], %[[VAL_18]]{{\[}}%[[VAL_7]]] : memref<3xindex>
// CHECK:           %[[VAL_20:.*]] = arith.cmpi ugt, %[[VAL_19]], %[[VAL_8]] : index
// CHECK:           %[[VAL_21:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_8]]] : memref<?xindex>
// CHECK:           %[[VAL_22:.*]] = arith.cmpi uge, %[[VAL_21]], %[[VAL_6]] : index
// CHECK:           %[[VAL_23:.*]] = arith.andi %[[VAL_20]], %[[VAL_22]] : i1
// CHECK:           %[[VAL_24:.*]] = arith.addi %[[VAL_21]], %[[VAL_3]] : index
// CHECK:           %[[VAL_25:.*]] = arith.select %[[VAL_23]], %[[VAL_24]], %[[VAL_8]] : index
// CHECK:           %[[VAL_26:.*]]:3 = scf.while (%[[VAL_27:.*]] = %[[VAL_20]], %[[VAL_28:.*]] = %[[VAL_21]], %[[VAL_29:.*]] = %[[VAL_25]], %[[VAL_30:.*]] = %[[VAL_11]]) : (i1, index, index, tensor<6x6xi32, #sparse>) -> (index, index, tensor<6x6xi32, #sparse>) {
// CHECK:             scf.condition(%[[VAL_27]]) %[[VAL_28]], %[[VAL_29]], %[[VAL_30]] : index, index, tensor<6x6xi32, #sparse>
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_31:.*]]: index, %[[VAL_32:.*]]: index, %[[VAL_33:.*]]: tensor<6x6xi32, #sparse>):
// CHECK:             %[[VAL_34:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_8]]] : memref<3xindex>
// CHECK:             %[[VAL_35:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_7]]] : memref<3xindex>
// CHECK:             memref.store %[[VAL_8]], %[[VAL_18]]{{\[}}%[[VAL_4]]] : memref<3xindex>
// CHECK:             %[[VAL_36:.*]] = arith.addi %[[VAL_32]], %[[VAL_6]] : index
// CHECK:             %[[VAL_37:.*]]:5 = scf.while (%[[VAL_38:.*]] = %[[VAL_34]], %[[VAL_39:.*]] = %[[VAL_10]], %[[VAL_40:.*]] = %[[VAL_5]], %[[VAL_41:.*]] = %[[VAL_8]], %[[VAL_42:.*]] = %[[VAL_8]]) : (index, i1, index, index, index) -> (index, i1, index, index, index) {
// CHECK:               %[[VAL_43:.*]] = arith.cmpi ult, %[[VAL_38]], %[[VAL_35]] : index
// CHECK:               %[[VAL_44:.*]] = scf.if %[[VAL_43]] -> (i1) {
// CHECK:                 %[[VAL_45:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_38]]] : memref<?xindex>
// CHECK:                 %[[VAL_46:.*]] = arith.cmpi ult, %[[VAL_45]], %[[VAL_36]] : index
// CHECK:                 scf.yield %[[VAL_46]] : i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_10]] : i1
// CHECK:               }
// CHECK:               scf.condition(%[[VAL_44]]) %[[VAL_38]], %[[VAL_39]], %[[VAL_40]], %[[VAL_41]], %[[VAL_42]] : index, i1, index, index, index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_47:.*]]: index, %[[VAL_48:.*]]: i1, %[[VAL_49:.*]]: index, %[[VAL_50:.*]]: index, %[[VAL_51:.*]]: index):
// CHECK:               %[[VAL_52:.*]] = arith.addi %[[VAL_47]], %[[VAL_7]] : index
// CHECK:               %[[VAL_53:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_47]]] : memref<?xindex>
// CHECK:               %[[VAL_54:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_52]]] : memref<?xindex>
// CHECK:               %[[VAL_55:.*]] = arith.cmpi ult, %[[VAL_53]], %[[VAL_54]] : index
// CHECK:               %[[VAL_56:.*]] = arith.ori %[[VAL_55]], %[[VAL_48]] : i1
// CHECK:               %[[VAL_57:.*]] = scf.if %[[VAL_55]] -> (index) {
// CHECK:                 %[[VAL_58:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_53]]] : memref<?xindex>
// CHECK:                 %[[VAL_59:.*]] = arith.cmpi ult, %[[VAL_58]], %[[VAL_49]] : index
// CHECK:                 %[[VAL_60:.*]] = arith.select %[[VAL_59]], %[[VAL_58]], %[[VAL_49]] : index
// CHECK:                 scf.yield %[[VAL_60]] : index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_49]] : index
// CHECK:               }
// CHECK:               memref.store %[[VAL_53]], %[[VAL_17]]{{\[}}%[[VAL_50]]] : memref<9xindex>
// CHECK:               %[[VAL_61:.*]] = arith.addi %[[VAL_50]], %[[VAL_6]] : index
// CHECK:               memref.store %[[VAL_54]], %[[VAL_17]]{{\[}}%[[VAL_61]]] : memref<9xindex>
// CHECK:               %[[VAL_62:.*]] = arith.addi %[[VAL_50]], %[[VAL_7]] : index
// CHECK:               %[[VAL_63:.*]] = arith.addi %[[VAL_51]], %[[VAL_7]] : index
// CHECK:               scf.yield %[[VAL_52]], %[[VAL_56]], %[[VAL_57]], %[[VAL_62]], %[[VAL_63]] : index, i1, index, index, index
// CHECK:             }
// CHECK:             %[[VAL_64:.*]] = arith.cmpi uge, %[[VAL_65:.*]]#2, %[[VAL_6]] : index
// CHECK:             %[[VAL_66:.*]] = arith.andi %[[VAL_65]]#1, %[[VAL_64]] : i1
// CHECK:             %[[VAL_67:.*]] = arith.addi %[[VAL_65]]#2, %[[VAL_3]] : index
// CHECK:             %[[VAL_68:.*]] = arith.select %[[VAL_66]], %[[VAL_67]], %[[VAL_8]] : index
// CHECK:             %[[VAL_69:.*]]:3 = scf.while (%[[VAL_70:.*]] = %[[VAL_65]]#1, %[[VAL_71:.*]] = %[[VAL_65]]#2, %[[VAL_72:.*]] = %[[VAL_68]], %[[VAL_73:.*]] = %[[VAL_33]]) : (i1, index, index, tensor<6x6xi32, #sparse>) -> (index, index, tensor<6x6xi32, #sparse>) {
// CHECK:               scf.condition(%[[VAL_70]]) %[[VAL_71]], %[[VAL_72]], %[[VAL_73]] : index, index, tensor<6x6xi32, #sparse>
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_74:.*]]: index, %[[VAL_75:.*]]: index, %[[VAL_76:.*]]: tensor<6x6xi32, #sparse>):
// CHECK:               %[[VAL_77:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_8]]] : memref<3xindex>
// CHECK:               %[[VAL_78:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_7]]] : memref<3xindex>
// CHECK:               %[[VAL_79:.*]]:3 = scf.while (%[[VAL_80:.*]] = %[[VAL_77]], %[[VAL_81:.*]] = %[[VAL_9]], %[[VAL_82:.*]] = %[[VAL_10]]) : (index, i32, i1) -> (index, i32, i1) {
// CHECK:                 %[[VAL_83:.*]] = arith.cmpi ult, %[[VAL_80]], %[[VAL_78]] : index
// CHECK:                 %[[VAL_84:.*]] = scf.if %[[VAL_83]] -> (i1) {
// CHECK:                   %[[VAL_85:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_80]]] : memref<?xindex>
// CHECK:                   %[[VAL_86:.*]] = arith.cmpi ult, %[[VAL_85]], %[[VAL_36]] : index
// CHECK:                   scf.yield %[[VAL_86]] : i1
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_10]] : i1
// CHECK:                 }
// CHECK:                 scf.condition(%[[VAL_84]]) %[[VAL_80]], %[[VAL_81]], %[[VAL_82]] : index, i32, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_87:.*]]: index, %[[VAL_88:.*]]: i32, %[[VAL_89:.*]]: i1):
// CHECK:                 %[[VAL_90:.*]] = arith.subi %[[VAL_87]], %[[VAL_77]] : index
// CHECK:                 %[[VAL_91:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_87]]] : memref<?xindex>
// CHECK:                 %[[VAL_92:.*]] = arith.subi %[[VAL_91]], %[[VAL_32]] : index
// CHECK:                 %[[VAL_93:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_90]]] : memref<9xindex>
// CHECK:                 %[[VAL_94:.*]] = arith.addi %[[VAL_90]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_95:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_94]]] : memref<9xindex>
// CHECK:                 %[[VAL_96:.*]] = arith.addi %[[VAL_75]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_97:.*]]:2 = scf.while (%[[VAL_98:.*]] = %[[VAL_93]], %[[VAL_99:.*]] = %[[VAL_88]]) : (index, i32) -> (index, i32) {
// CHECK:                   %[[VAL_100:.*]] = arith.cmpi ult, %[[VAL_98]], %[[VAL_95]] : index
// CHECK:                   %[[VAL_101:.*]] = scf.if %[[VAL_100]] -> (i1) {
// CHECK:                     %[[VAL_102:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_98]]] : memref<?xindex>
// CHECK:                     %[[VAL_103:.*]] = arith.cmpi ult, %[[VAL_102]], %[[VAL_96]] : index
// CHECK:                     scf.yield %[[VAL_103]] : i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_10]] : i1
// CHECK:                   }
// CHECK:                   scf.condition(%[[VAL_101]]) %[[VAL_98]], %[[VAL_99]] : index, i32
// CHECK:                 } do {
// CHECK:                 ^bb0(%[[VAL_104:.*]]: index, %[[VAL_105:.*]]: i32):
// CHECK:                   %[[VAL_106:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_104]]] : memref<?xindex>
// CHECK:                   %[[VAL_107:.*]] = arith.subi %[[VAL_106]], %[[VAL_75]] : index
// CHECK:                   %[[VAL_108:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_104]]] : memref<?xi32>
// CHECK:                   %[[VAL_109:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_92]], %[[VAL_107]]] : tensor<3x3xi32>
// CHECK:                   %[[VAL_110:.*]] = arith.muli %[[VAL_108]], %[[VAL_109]] : i32
// CHECK:                   %[[VAL_111:.*]] = arith.addi %[[VAL_105]], %[[VAL_110]] : i32
// CHECK:                   %[[VAL_112:.*]] = arith.addi %[[VAL_104]], %[[VAL_7]] : index
// CHECK:                   scf.yield %[[VAL_112]], %[[VAL_111]] : index, i32
// CHECK:                 }
// CHECK:                 %[[VAL_113:.*]] = arith.addi %[[VAL_87]], %[[VAL_7]] : index
// CHECK:                 scf.yield %[[VAL_113]], %[[VAL_114:.*]]#1, %[[VAL_2]] : index, i32, i1
// CHECK:               }
// CHECK:               %[[VAL_115:.*]] = scf.if %[[VAL_116:.*]]#2 -> (tensor<6x6xi32, #sparse>) {
// CHECK:                 %[[VAL_117:.*]] = sparse_tensor.insert %[[VAL_116]]#1 into %[[VAL_76]]{{\[}}%[[VAL_32]], %[[VAL_75]]] : tensor<6x6xi32, #sparse>
// CHECK:                 scf.yield %[[VAL_117]] : tensor<6x6xi32, #sparse>
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_76]] : tensor<6x6xi32, #sparse>
// CHECK:               }
// CHECK:               %[[VAL_118:.*]] = arith.cmpi ugt, %[[VAL_74]], %[[VAL_75]] : index
// CHECK:               %[[VAL_119:.*]]:3 = scf.if %[[VAL_118]] -> (index, i1, index) {
// CHECK:                 %[[VAL_120:.*]] = arith.addi %[[VAL_75]], %[[VAL_7]] : index
// CHECK:                 scf.yield %[[VAL_74]], %[[VAL_2]], %[[VAL_120]] : index, i1, index
// CHECK:               } else {
// CHECK:                 %[[VAL_121:.*]]:2 = scf.for %[[VAL_122:.*]] = %[[VAL_8]] to %[[VAL_65]]#3 step %[[VAL_7]] iter_args(%[[VAL_123:.*]] = %[[VAL_5]], %[[VAL_124:.*]] = %[[VAL_10]]) -> (index, i1) {
// CHECK:                   %[[VAL_125:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_122]]] : memref<9xindex>
// CHECK:                   %[[VAL_126:.*]] = arith.addi %[[VAL_122]], %[[VAL_6]] : index
// CHECK:                   %[[VAL_127:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_126]]] : memref<9xindex>
// CHECK:                   %[[VAL_128:.*]] = arith.cmpi ult, %[[VAL_125]], %[[VAL_127]] : index
// CHECK:                   %[[VAL_129:.*]] = scf.if %[[VAL_128]] -> (index) {
// CHECK:                     %[[VAL_130:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_125]]] : memref<?xindex>
// CHECK:                     %[[VAL_131:.*]] = arith.cmpi eq, %[[VAL_130]], %[[VAL_74]] : index
// CHECK:                     %[[VAL_132:.*]] = scf.if %[[VAL_131]] -> (index) {
// CHECK:                       %[[VAL_133:.*]] = arith.addi %[[VAL_125]], %[[VAL_7]] : index
// CHECK:                       memref.store %[[VAL_133]], %[[VAL_17]]{{\[}}%[[VAL_122]]] : memref<9xindex>
// CHECK:                       scf.yield %[[VAL_133]] : index
// CHECK:                     } else {
// CHECK:                       scf.yield %[[VAL_125]] : index
// CHECK:                     }
// CHECK:                     scf.yield %[[VAL_132]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_125]] : index
// CHECK:                   }
// CHECK:                   %[[VAL_134:.*]] = arith.cmpi ult, %[[VAL_129]], %[[VAL_127]] : index
// CHECK:                   %[[VAL_135:.*]] = scf.if %[[VAL_134]] -> (index) {
// CHECK:                     %[[VAL_136:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_129]]] : memref<?xindex>
// CHECK:                     scf.yield %[[VAL_136]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_123]] : index
// CHECK:                   }
// CHECK:                   %[[VAL_137:.*]] = arith.ori %[[VAL_134]], %[[VAL_124]] : i1
// CHECK:                   %[[VAL_138:.*]] = arith.cmpi ult, %[[VAL_135]], %[[VAL_123]] : index
// CHECK:                   %[[VAL_139:.*]] = arith.select %[[VAL_138]], %[[VAL_135]], %[[VAL_123]] : index
// CHECK:                   scf.yield %[[VAL_139]], %[[VAL_137]] : index, i1
// CHECK:                 }
// CHECK:                 %[[VAL_140:.*]] = arith.addi %[[VAL_141:.*]]#0, %[[VAL_7]] : index
// CHECK:                 %[[VAL_142:.*]] = arith.addi %[[VAL_141]]#0, %[[VAL_3]] : index
// CHECK:                 %[[VAL_143:.*]] = arith.cmpi uge, %[[VAL_140]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_144:.*]] = arith.select %[[VAL_143]], %[[VAL_142]], %[[VAL_8]] : index
// CHECK:                 scf.yield %[[VAL_141]]#0, %[[VAL_141]]#1, %[[VAL_144]] : index, i1, index
// CHECK:               }
// CHECK:               %[[VAL_145:.*]] = arith.addi %[[VAL_75]], %[[VAL_7]] : index
// CHECK:               %[[VAL_146:.*]] = arith.cmpi ugt, %[[VAL_147:.*]]#2, %[[VAL_145]] : index
// CHECK:               %[[VAL_148:.*]] = arith.select %[[VAL_146]], %[[VAL_147]]#2, %[[VAL_145]] : index
// CHECK:               %[[VAL_149:.*]] = arith.addi %[[VAL_148]], %[[VAL_6]] : index
// CHECK:               %[[VAL_150:.*]] = arith.cmpi ule, %[[VAL_149]], %[[VAL_5]] : index
// CHECK:               %[[VAL_151:.*]] = arith.andi %[[VAL_147]]#1, %[[VAL_150]] : i1
// CHECK:               scf.yield %[[VAL_151]], %[[VAL_147]]#0, %[[VAL_148]], %[[VAL_115]] : i1, index, index, tensor<6x6xi32, #sparse>
// CHECK:             }
// CHECK:             %[[VAL_152:.*]] = arith.cmpi ugt, %[[VAL_31]], %[[VAL_32]] : index
// CHECK:             %[[VAL_153:.*]]:3 = scf.if %[[VAL_152]] -> (index, i1, index) {
// CHECK:               %[[VAL_154:.*]] = arith.addi %[[VAL_32]], %[[VAL_7]] : index
// CHECK:               scf.yield %[[VAL_31]], %[[VAL_2]], %[[VAL_154]] : index, i1, index
// CHECK:             } else {
// CHECK:               %[[VAL_155:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_8]]] : memref<3xindex>
// CHECK:               %[[VAL_156:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_7]]] : memref<3xindex>
// CHECK:               %[[VAL_157:.*]] = arith.cmpi ult, %[[VAL_155]], %[[VAL_156]] : index
// CHECK:               %[[VAL_158:.*]] = scf.if %[[VAL_157]] -> (index) {
// CHECK:                 %[[VAL_159:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_155]]] : memref<?xindex>
// CHECK:                 %[[VAL_160:.*]] = arith.cmpi eq, %[[VAL_159]], %[[VAL_31]] : index
// CHECK:                 %[[VAL_161:.*]] = scf.if %[[VAL_160]] -> (index) {
// CHECK:                   %[[VAL_162:.*]] = arith.addi %[[VAL_155]], %[[VAL_7]] : index
// CHECK:                   memref.store %[[VAL_162]], %[[VAL_18]]{{\[}}%[[VAL_8]]] : memref<3xindex>
// CHECK:                   scf.yield %[[VAL_162]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_155]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_161]] : index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_155]] : index
// CHECK:               }
// CHECK:               %[[VAL_163:.*]] = arith.cmpi ult, %[[VAL_158]], %[[VAL_156]] : index
// CHECK:               %[[VAL_164:.*]] = scf.if %[[VAL_163]] -> (index) {
// CHECK:                 %[[VAL_165:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_158]]] : memref<?xindex>
// CHECK:                 scf.yield %[[VAL_165]] : index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_5]] : index
// CHECK:               }
// CHECK:               %[[VAL_166:.*]] = arith.cmpi ult, %[[VAL_164]], %[[VAL_5]] : index
// CHECK:               %[[VAL_167:.*]] = arith.select %[[VAL_166]], %[[VAL_164]], %[[VAL_5]] : index
// CHECK:               %[[VAL_168:.*]] = arith.addi %[[VAL_167]], %[[VAL_7]] : index
// CHECK:               %[[VAL_169:.*]] = arith.addi %[[VAL_167]], %[[VAL_3]] : index
// CHECK:               %[[VAL_170:.*]] = arith.cmpi uge, %[[VAL_168]], %[[VAL_6]] : index
// CHECK:               %[[VAL_171:.*]] = arith.select %[[VAL_170]], %[[VAL_169]], %[[VAL_8]] : index
// CHECK:               scf.yield %[[VAL_167]], %[[VAL_163]], %[[VAL_171]] : index, i1, index
// CHECK:             }
// CHECK:             %[[VAL_172:.*]] = arith.addi %[[VAL_32]], %[[VAL_7]] : index
// CHECK:             %[[VAL_173:.*]] = arith.cmpi ugt, %[[VAL_174:.*]]#2, %[[VAL_172]] : index
// CHECK:             %[[VAL_175:.*]] = arith.select %[[VAL_173]], %[[VAL_174]]#2, %[[VAL_172]] : index
// CHECK:             %[[VAL_176:.*]] = arith.addi %[[VAL_175]], %[[VAL_6]] : index
// CHECK:             %[[VAL_177:.*]] = arith.cmpi ule, %[[VAL_176]], %[[VAL_5]] : index
// CHECK:             %[[VAL_178:.*]] = arith.andi %[[VAL_174]]#1, %[[VAL_177]] : i1
// CHECK:             scf.yield %[[VAL_178]], %[[VAL_174]]#0, %[[VAL_175]], %[[VAL_179:.*]]#2 : i1, index, index, tensor<6x6xi32, #sparse>
// CHECK:           }
// CHECK:           %[[VAL_180:.*]] = sparse_tensor.load %[[VAL_181:.*]]#2 hasInserts : tensor<6x6xi32, #sparse>
// CHECK:           return %[[VAL_180]] : tensor<6x6xi32, #sparse>
// CHECK:         }
func.func @conv2d_all_sparse_CSR(%arg0: tensor<8x8xi32, #DCSR>,
                                 %arg1: tensor<3x3xi32>) -> tensor<6x6xi32, #DCSR> {
  %0 = tensor.empty() : tensor<6x6xi32, #DCSR>
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
