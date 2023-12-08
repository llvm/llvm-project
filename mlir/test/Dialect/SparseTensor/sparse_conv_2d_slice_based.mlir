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
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 5 : index
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_11:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_12:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_13:.*]] = tensor.empty() : tensor<6x6xi32, #sparse>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<8x8xi32, #sparse> to memref<?xindex>
// CHECK-DAG:       %[[VAL_15:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<8x8xi32, #sparse> to memref<?xindex>
// CHECK-DAG:       %[[VAL_16:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<8x8xi32, #sparse> to memref<?xindex>
// CHECK-DAG:       %[[VAL_17:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<8x8xi32, #sparse> to memref<?xindex>
// CHECK-DAG:       %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x8xi32, #sparse> to memref<?xi32>
// CHECK:           %[[VAL_19:.*]] = memref.alloca() : memref<11xindex>
// CHECK:           %[[VAL_20:.*]] = memref.alloca() : memref<5xindex>
// CHECK:           %[[VAL_21:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_7]]] : memref<?xindex>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_20]]{{\[}}%[[VAL_7]]] : memref<5xindex>
// CHECK:           memref.store %[[VAL_10]], %[[VAL_20]]{{\[}}%[[VAL_9]]] : memref<5xindex>
// CHECK:           memref.store %[[VAL_21]], %[[VAL_20]]{{\[}}%[[VAL_6]]] : memref<5xindex>
// CHECK:           %[[VAL_22:.*]] = arith.cmpi ugt, %[[VAL_21]], %[[VAL_10]] : index
// CHECK:           %[[VAL_23:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK:           %[[VAL_24:.*]] = arith.cmpi uge, %[[VAL_23]], %[[VAL_6]] : index
// CHECK:           %[[VAL_25:.*]] = arith.andi %[[VAL_22]], %[[VAL_24]] : i1
// CHECK:           %[[VAL_26:.*]] = arith.addi %[[VAL_23]], %[[VAL_3]] : index
// CHECK:           %[[VAL_27:.*]] = arith.select %[[VAL_25]], %[[VAL_26]], %[[VAL_10]] : index
// CHECK:           %[[VAL_28:.*]]:3 = scf.while (%[[VAL_29:.*]] = %[[VAL_22]], %[[VAL_30:.*]] = %[[VAL_23]], %[[VAL_31:.*]] = %[[VAL_27]], %[[VAL_32:.*]] = %[[VAL_13]]) : (i1, index, index, tensor<6x6xi32, #sparse>) -> (index, index, tensor<6x6xi32, #sparse>) {
// CHECK:             scf.condition(%[[VAL_29]]) %[[VAL_30]], %[[VAL_31]], %[[VAL_32]] : index, index, tensor<6x6xi32, #sparse>
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_33:.*]]: index, %[[VAL_34:.*]]: index, %[[VAL_35:.*]]: tensor<6x6xi32, #sparse>):
// CHECK:             %[[VAL_36:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_9]]] : memref<5xindex>
// CHECK:             %[[VAL_37:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_6]]] : memref<5xindex>
// CHECK:             memref.store %[[VAL_10]], %[[VAL_20]]{{\[}}%[[VAL_4]]] : memref<5xindex>
// CHECK:             %[[VAL_38:.*]] = arith.addi %[[VAL_34]], %[[VAL_6]] : index
// CHECK:             %[[VAL_39:.*]]:5 = scf.while (%[[VAL_40:.*]] = %[[VAL_36]], %[[VAL_41:.*]] = %[[VAL_12]], %[[VAL_42:.*]] = %[[VAL_5]], %[[VAL_43:.*]] = %[[VAL_10]], %[[VAL_44:.*]] = %[[VAL_10]]) : (index, i1, index, index, index) -> (index, i1, index, index, index) {
// CHECK:               %[[VAL_45:.*]] = arith.cmpi ult, %[[VAL_40]], %[[VAL_37]] : index
// CHECK:               %[[VAL_46:.*]] = scf.if %[[VAL_45]] -> (i1) {
// CHECK:                 %[[VAL_47:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_40]]] : memref<?xindex>
// CHECK:                 %[[VAL_48:.*]] = arith.cmpi ult, %[[VAL_47]], %[[VAL_38]] : index
// CHECK:                 scf.yield %[[VAL_48]] : i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_12]] : i1
// CHECK:               }
// CHECK:               scf.condition(%[[VAL_46]]) %[[VAL_40]], %[[VAL_41]], %[[VAL_42]], %[[VAL_43]], %[[VAL_44]] : index, i1, index, index, index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_49:.*]]: index, %[[VAL_50:.*]]: i1, %[[VAL_51:.*]]: index, %[[VAL_52:.*]]: index, %[[VAL_53:.*]]: index):
// CHECK:               %[[VAL_54:.*]] = arith.addi %[[VAL_49]], %[[VAL_7]] : index
// CHECK:               %[[VAL_55:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_49]]] : memref<?xindex>
// CHECK:               %[[VAL_56:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_54]]] : memref<?xindex>
// CHECK:               %[[VAL_57:.*]] = arith.cmpi ult, %[[VAL_55]], %[[VAL_56]] : index
// CHECK:               %[[VAL_58:.*]] = arith.ori %[[VAL_57]], %[[VAL_50]] : i1
// CHECK:               %[[VAL_59:.*]] = scf.if %[[VAL_57]] -> (index) {
// CHECK:                 %[[VAL_60:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_55]]] : memref<?xindex>
// CHECK:                 %[[VAL_61:.*]] = arith.cmpi ult, %[[VAL_60]], %[[VAL_51]] : index
// CHECK:                 %[[VAL_62:.*]] = arith.select %[[VAL_61]], %[[VAL_60]], %[[VAL_51]] : index
// CHECK:                 scf.yield %[[VAL_62]] : index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_51]] : index
// CHECK:               }
// CHECK:               %[[VAL_63:.*]] = arith.addi %[[VAL_52]], %[[VAL_9]] : index
// CHECK:               memref.store %[[VAL_55]], %[[VAL_19]]{{\[}}%[[VAL_63]]] : memref<11xindex>
// CHECK:               %[[VAL_64:.*]] = arith.addi %[[VAL_52]], %[[VAL_8]] : index
// CHECK:               memref.store %[[VAL_56]], %[[VAL_19]]{{\[}}%[[VAL_64]]] : memref<11xindex>
// CHECK:               %[[VAL_65:.*]] = arith.addi %[[VAL_52]], %[[VAL_7]] : index
// CHECK:               %[[VAL_66:.*]] = arith.addi %[[VAL_53]], %[[VAL_7]] : index
// CHECK:               scf.yield %[[VAL_54]], %[[VAL_58]], %[[VAL_59]], %[[VAL_65]], %[[VAL_66]] : index, i1, index, index, index
// CHECK:             }
// CHECK:             memref.store %[[VAL_10]], %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<11xindex>
// CHECK:             %[[VAL_67:.*]] = arith.cmpi uge, %[[VAL_68:.*]]#2, %[[VAL_6]] : index
// CHECK:             %[[VAL_69:.*]] = arith.andi %[[VAL_68]]#1, %[[VAL_67]] : i1
// CHECK:             %[[VAL_70:.*]] = arith.addi %[[VAL_68]]#2, %[[VAL_3]] : index
// CHECK:             %[[VAL_71:.*]] = arith.select %[[VAL_69]], %[[VAL_70]], %[[VAL_10]] : index
// CHECK:             %[[VAL_72:.*]]:3 = scf.while (%[[VAL_73:.*]] = %[[VAL_68]]#1, %[[VAL_74:.*]] = %[[VAL_68]]#2, %[[VAL_75:.*]] = %[[VAL_71]], %[[VAL_76:.*]] = %[[VAL_35]]) : (i1, index, index, tensor<6x6xi32, #sparse>) -> (index, index, tensor<6x6xi32, #sparse>) {
// CHECK:               scf.condition(%[[VAL_73]]) %[[VAL_74]], %[[VAL_75]], %[[VAL_76]] : index, index, tensor<6x6xi32, #sparse>
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_77:.*]]: index, %[[VAL_78:.*]]: index, %[[VAL_79:.*]]: tensor<6x6xi32, #sparse>):
// CHECK:               %[[VAL_80:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_7]]] : memref<5xindex>
// CHECK:               %[[VAL_81:.*]] = arith.addi %[[VAL_80]], %[[VAL_9]] : index
// CHECK:               %[[VAL_82:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_81]]] : memref<5xindex>
// CHECK:               %[[VAL_83:.*]] = arith.addi %[[VAL_80]], %[[VAL_6]] : index
// CHECK:               %[[VAL_84:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_83]]] : memref<5xindex>
// CHECK:               %[[VAL_85:.*]]:3 = scf.while (%[[VAL_86:.*]] = %[[VAL_82]], %[[VAL_87:.*]] = %[[VAL_11]], %[[VAL_88:.*]] = %[[VAL_12]]) : (index, i32, i1) -> (index, i32, i1) {
// CHECK:                 %[[VAL_89:.*]] = arith.cmpi ult, %[[VAL_86]], %[[VAL_84]] : index
// CHECK:                 %[[VAL_90:.*]] = scf.if %[[VAL_89]] -> (i1) {
// CHECK:                   %[[VAL_91:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_86]]] : memref<?xindex>
// CHECK:                   %[[VAL_92:.*]] = arith.cmpi ult, %[[VAL_91]], %[[VAL_38]] : index
// CHECK:                   scf.yield %[[VAL_92]] : i1
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_12]] : i1
// CHECK:                 }
// CHECK:                 scf.condition(%[[VAL_90]]) %[[VAL_86]], %[[VAL_87]], %[[VAL_88]] : index, i32, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_93:.*]]: index, %[[VAL_94:.*]]: i32, %[[VAL_95:.*]]: i1):
// CHECK:                 %[[VAL_96:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_93]]] : memref<?xindex>
// CHECK:                 %[[VAL_97:.*]] = arith.subi %[[VAL_96]], %[[VAL_34]] : index
// CHECK:                 %[[VAL_98:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<11xindex>
// CHECK:                 %[[VAL_99:.*]] = arith.addi %[[VAL_98]], %[[VAL_9]] : index
// CHECK:                 %[[VAL_100:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_99]]] : memref<11xindex>
// CHECK:                 %[[VAL_101:.*]] = arith.addi %[[VAL_98]], %[[VAL_8]] : index
// CHECK:                 %[[VAL_102:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_101]]] : memref<11xindex>
// CHECK:                 %[[VAL_103:.*]] = arith.addi %[[VAL_78]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_104:.*]]:2 = scf.while (%[[VAL_105:.*]] = %[[VAL_100]], %[[VAL_106:.*]] = %[[VAL_94]]) : (index, i32) -> (index, i32) {
// CHECK:                   %[[VAL_107:.*]] = arith.cmpi ult, %[[VAL_105]], %[[VAL_102]] : index
// CHECK:                   %[[VAL_108:.*]] = scf.if %[[VAL_107]] -> (i1) {
// CHECK:                     %[[VAL_109:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_105]]] : memref<?xindex>
// CHECK:                     %[[VAL_110:.*]] = arith.cmpi ult, %[[VAL_109]], %[[VAL_103]] : index
// CHECK:                     scf.yield %[[VAL_110]] : i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_12]] : i1
// CHECK:                   }
// CHECK:                   scf.condition(%[[VAL_108]]) %[[VAL_105]], %[[VAL_106]] : index, i32
// CHECK:                 } do {
// CHECK:                 ^bb0(%[[VAL_111:.*]]: index, %[[VAL_112:.*]]: i32):
// CHECK:                   %[[VAL_113:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_111]]] : memref<?xindex>
// CHECK:                   %[[VAL_114:.*]] = arith.subi %[[VAL_113]], %[[VAL_78]] : index
// CHECK:                   %[[VAL_115:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_111]]] : memref<?xi32>
// CHECK:                   %[[VAL_116:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_97]], %[[VAL_114]]] : tensor<3x3xi32>
// CHECK:                   %[[VAL_117:.*]] = arith.muli %[[VAL_115]], %[[VAL_116]] : i32
// CHECK:                   %[[VAL_118:.*]] = arith.addi %[[VAL_112]], %[[VAL_117]] : i32
// CHECK:                   %[[VAL_119:.*]] = arith.addi %[[VAL_111]], %[[VAL_7]] : index
// CHECK:                   scf.yield %[[VAL_119]], %[[VAL_118]] : index, i32
// CHECK:                 }
// CHECK:                 %[[VAL_120:.*]] = arith.addi %[[VAL_93]], %[[VAL_7]] : index
// CHECK:                 %[[VAL_121:.*]] = arith.addi %[[VAL_98]], %[[VAL_7]] : index
// CHECK:                 memref.store %[[VAL_121]], %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<11xindex>
// CHECK:                 scf.yield %[[VAL_120]], %[[VAL_122:.*]]#1, %[[VAL_2]] : index, i32, i1
// CHECK:               }
// CHECK:               %[[VAL_123:.*]] = scf.if %[[VAL_124:.*]]#2 -> (tensor<6x6xi32, #sparse>) {
// CHECK:                 %[[VAL_125:.*]] = sparse_tensor.insert %[[VAL_124]]#1 into %[[VAL_79]]{{\[}}%[[VAL_34]], %[[VAL_78]]] : tensor<6x6xi32, #sparse>
// CHECK:                 scf.yield %[[VAL_125]] : tensor<6x6xi32, #sparse>
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_79]] : tensor<6x6xi32, #sparse>
// CHECK:               }
// CHECK:               memref.store %[[VAL_10]], %[[VAL_20]]{{\[}}%[[VAL_7]]] : memref<5xindex>
// CHECK:               memref.store %[[VAL_10]], %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<11xindex>
// CHECK:               %[[VAL_126:.*]] = arith.cmpi ugt, %[[VAL_77]], %[[VAL_78]] : index
// CHECK:               %[[VAL_127:.*]]:3 = scf.if %[[VAL_126]] -> (index, i1, index) {
// CHECK:                 %[[VAL_128:.*]] = arith.addi %[[VAL_78]], %[[VAL_7]] : index
// CHECK:                 scf.yield %[[VAL_77]], %[[VAL_2]], %[[VAL_128]] : index, i1, index
// CHECK:               } else {
// CHECK:                 %[[VAL_129:.*]]:2 = scf.for %[[VAL_130:.*]] = %[[VAL_10]] to %[[VAL_68]]#3 step %[[VAL_7]] iter_args(%[[VAL_131:.*]] = %[[VAL_5]], %[[VAL_132:.*]] = %[[VAL_12]]) -> (index, i1) {
// CHECK:                   %[[VAL_133:.*]] = arith.addi %[[VAL_130]], %[[VAL_9]] : index
// CHECK:                   %[[VAL_134:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_133]]] : memref<11xindex>
// CHECK:                   %[[VAL_135:.*]] = arith.addi %[[VAL_130]], %[[VAL_8]] : index
// CHECK:                   %[[VAL_136:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_135]]] : memref<11xindex>
// CHECK:                   %[[VAL_137:.*]] = arith.cmpi ult, %[[VAL_134]], %[[VAL_136]] : index
// CHECK:                   %[[VAL_138:.*]] = scf.if %[[VAL_137]] -> (index) {
// CHECK:                     %[[VAL_139:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_134]]] : memref<?xindex>
// CHECK:                     %[[VAL_140:.*]] = arith.cmpi eq, %[[VAL_139]], %[[VAL_77]] : index
// CHECK:                     %[[VAL_141:.*]] = scf.if %[[VAL_140]] -> (index) {
// CHECK:                       %[[VAL_142:.*]] = arith.addi %[[VAL_134]], %[[VAL_7]] : index
// CHECK:                       memref.store %[[VAL_142]], %[[VAL_19]]{{\[}}%[[VAL_133]]] : memref<11xindex>
// CHECK:                       scf.yield %[[VAL_142]] : index
// CHECK:                     } else {
// CHECK:                       scf.yield %[[VAL_134]] : index
// CHECK:                     }
// CHECK:                     scf.yield %[[VAL_141]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_134]] : index
// CHECK:                   }
// CHECK:                   %[[VAL_143:.*]] = arith.cmpi ult, %[[VAL_138]], %[[VAL_136]] : index
// CHECK:                   %[[VAL_144:.*]] = scf.if %[[VAL_143]] -> (index) {
// CHECK:                     %[[VAL_145:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_138]]] : memref<?xindex>
// CHECK:                     scf.yield %[[VAL_145]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_131]] : index
// CHECK:                   }
// CHECK:                   %[[VAL_146:.*]] = arith.ori %[[VAL_143]], %[[VAL_132]] : i1
// CHECK:                   %[[VAL_147:.*]] = arith.cmpi ult, %[[VAL_144]], %[[VAL_131]] : index
// CHECK:                   %[[VAL_148:.*]] = arith.select %[[VAL_147]], %[[VAL_144]], %[[VAL_131]] : index
// CHECK:                   scf.yield %[[VAL_148]], %[[VAL_146]] : index, i1
// CHECK:                 }
// CHECK:                 %[[VAL_149:.*]] = arith.addi %[[VAL_150:.*]]#0, %[[VAL_7]] : index
// CHECK:                 %[[VAL_151:.*]] = arith.addi %[[VAL_150]]#0, %[[VAL_3]] : index
// CHECK:                 %[[VAL_152:.*]] = arith.cmpi uge, %[[VAL_149]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_153:.*]] = arith.select %[[VAL_152]], %[[VAL_151]], %[[VAL_10]] : index
// CHECK:                 scf.yield %[[VAL_150]]#0, %[[VAL_150]]#1, %[[VAL_153]] : index, i1, index
// CHECK:               }
// CHECK:               %[[VAL_154:.*]] = arith.addi %[[VAL_78]], %[[VAL_7]] : index
// CHECK:               %[[VAL_155:.*]] = arith.cmpi ugt, %[[VAL_156:.*]]#2, %[[VAL_154]] : index
// CHECK:               %[[VAL_157:.*]] = arith.select %[[VAL_155]], %[[VAL_156]]#2, %[[VAL_154]] : index
// CHECK:               %[[VAL_158:.*]] = arith.addi %[[VAL_157]], %[[VAL_6]] : index
// CHECK:               %[[VAL_159:.*]] = arith.cmpi ule, %[[VAL_158]], %[[VAL_5]] : index
// CHECK:               %[[VAL_160:.*]] = arith.andi %[[VAL_156]]#1, %[[VAL_159]] : i1
// CHECK:               scf.yield %[[VAL_160]], %[[VAL_156]]#0, %[[VAL_157]], %[[VAL_123]] : i1, index, index, tensor<6x6xi32, #sparse>
// CHECK:             }
// CHECK:             memref.store %[[VAL_10]], %[[VAL_20]]{{\[}}%[[VAL_7]]] : memref<5xindex>
// CHECK:             %[[VAL_161:.*]] = arith.cmpi ugt, %[[VAL_33]], %[[VAL_34]] : index
// CHECK:             %[[VAL_162:.*]]:3 = scf.if %[[VAL_161]] -> (index, i1, index) {
// CHECK:               %[[VAL_163:.*]] = arith.addi %[[VAL_34]], %[[VAL_7]] : index
// CHECK:               scf.yield %[[VAL_33]], %[[VAL_2]], %[[VAL_163]] : index, i1, index
// CHECK:             } else {
// CHECK:               %[[VAL_164:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_9]]] : memref<5xindex>
// CHECK:               %[[VAL_165:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_6]]] : memref<5xindex>
// CHECK:               %[[VAL_166:.*]] = arith.cmpi ult, %[[VAL_164]], %[[VAL_165]] : index
// CHECK:               %[[VAL_167:.*]] = scf.if %[[VAL_166]] -> (index) {
// CHECK:                 %[[VAL_168:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_164]]] : memref<?xindex>
// CHECK:                 %[[VAL_169:.*]] = arith.cmpi eq, %[[VAL_168]], %[[VAL_33]] : index
// CHECK:                 %[[VAL_170:.*]] = scf.if %[[VAL_169]] -> (index) {
// CHECK:                   %[[VAL_171:.*]] = arith.addi %[[VAL_164]], %[[VAL_7]] : index
// CHECK:                   memref.store %[[VAL_171]], %[[VAL_20]]{{\[}}%[[VAL_9]]] : memref<5xindex>
// CHECK:                   scf.yield %[[VAL_171]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_164]] : index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_170]] : index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_164]] : index
// CHECK:               }
// CHECK:               %[[VAL_172:.*]] = arith.cmpi ult, %[[VAL_167]], %[[VAL_165]] : index
// CHECK:               %[[VAL_173:.*]] = scf.if %[[VAL_172]] -> (index) {
// CHECK:                 %[[VAL_174:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_167]]] : memref<?xindex>
// CHECK:                 scf.yield %[[VAL_174]] : index
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_5]] : index
// CHECK:               }
// CHECK:               %[[VAL_175:.*]] = arith.cmpi ult, %[[VAL_173]], %[[VAL_5]] : index
// CHECK:               %[[VAL_176:.*]] = arith.select %[[VAL_175]], %[[VAL_173]], %[[VAL_5]] : index
// CHECK:               %[[VAL_177:.*]] = arith.addi %[[VAL_176]], %[[VAL_7]] : index
// CHECK:               %[[VAL_178:.*]] = arith.addi %[[VAL_176]], %[[VAL_3]] : index
// CHECK:               %[[VAL_179:.*]] = arith.cmpi uge, %[[VAL_177]], %[[VAL_6]] : index
// CHECK:               %[[VAL_180:.*]] = arith.select %[[VAL_179]], %[[VAL_178]], %[[VAL_10]] : index
// CHECK:               scf.yield %[[VAL_176]], %[[VAL_172]], %[[VAL_180]] : index, i1, index
// CHECK:             }
// CHECK:             %[[VAL_181:.*]] = arith.addi %[[VAL_34]], %[[VAL_7]] : index
// CHECK:             %[[VAL_182:.*]] = arith.cmpi ugt, %[[VAL_183:.*]]#2, %[[VAL_181]] : index
// CHECK:             %[[VAL_184:.*]] = arith.select %[[VAL_182]], %[[VAL_183]]#2, %[[VAL_181]] : index
// CHECK:             %[[VAL_185:.*]] = arith.addi %[[VAL_184]], %[[VAL_6]] : index
// CHECK:             %[[VAL_186:.*]] = arith.cmpi ule, %[[VAL_185]], %[[VAL_5]] : index
// CHECK:             %[[VAL_187:.*]] = arith.andi %[[VAL_183]]#1, %[[VAL_186]] : i1
// CHECK:             scf.yield %[[VAL_187]], %[[VAL_183]]#0, %[[VAL_184]], %[[VAL_188:.*]]#2 : i1, index, index, tensor<6x6xi32, #sparse>
// CHECK:           }
// CHECK:           %[[VAL_189:.*]] = sparse_tensor.load %[[VAL_190:.*]]#2 hasInserts : tensor<6x6xi32, #sparse>
// CHECK:           return %[[VAL_189]] : tensor<6x6xi32, #sparse>
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
