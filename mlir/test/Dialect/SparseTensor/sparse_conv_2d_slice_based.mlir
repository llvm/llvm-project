// RUN: mlir-opt %s --sparse-reinterpret-map --sparsification --canonicalize --cse | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#DCSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>


// CHECK-LABEL:   func.func @conv2d_all_sparse_CSR(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x8xi32, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<3x3xi32>) -> tensor<6x6xi32, #sparse{{[0-9]*}}>
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
// CHECK-DAG:       %[[VAL_13:.*]] = tensor.empty() : tensor<6x6xi32, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<8x8xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_15:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<8x8xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_16:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<8x8xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_17:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<8x8xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_18:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x8xi32, #sparse{{[0-9]*}}> to memref<?xi32>
// CHECK-DAG:       %[[VAL_19:.*]] = memref.alloca() : memref<11xindex>
// CHECK-DAG:       %[[VAL_20:.*]] = memref.alloca() : memref<5xindex>
// CHECK-DAG:       %[[VAL_21:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_7]]] : memref<?xindex>
// CHECK-DAG:       memref.store %[[VAL_7]], %[[VAL_20]]{{\[}}%[[VAL_10]]] : memref<5xindex>
// CHECK-DAG:       memref.store %[[VAL_10]], %[[VAL_20]]{{\[}}%[[VAL_7]]] : memref<5xindex>
// CHECK-DAG:       memref.store %[[VAL_10]], %[[VAL_20]]{{\[}}%[[VAL_9]]] : memref<5xindex>
// CHECK-DAG:       memref.store %[[VAL_21]], %[[VAL_20]]{{\[}}%[[VAL_6]]] : memref<5xindex>
// CHECK:           %[[VAL_22:.*]] = arith.cmpi ugt, %[[VAL_21]], %[[VAL_10]] : index
// CHECK:           %[[VAL_23:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK:           %[[VAL_24:.*]] = arith.cmpi uge, %[[VAL_23]], %[[VAL_6]] : index
// CHECK:           %[[VAL_25:.*]] = arith.andi %[[VAL_22]], %[[VAL_24]] : i1
// CHECK:           %[[VAL_26:.*]] = arith.addi %[[VAL_23]], %[[VAL_3]] : index
// CHECK:           %[[VAL_27:.*]] = arith.select %[[VAL_25]], %[[VAL_26]], %[[VAL_10]] : index
// CHECK:           %[[VAL_28:.*]]:3 = scf.while (%[[VAL_29:.*]] = %[[VAL_22]], %[[VAL_30:.*]] = %[[VAL_23]], %[[VAL_31:.*]] = %[[VAL_27]], %[[VAL_32:.*]] = %[[VAL_13]]) : (i1, index, index, tensor<6x6xi32, #sparse{{[0-9]*}}>) -> (index, index, tensor<6x6xi32, #sparse{{[0-9]*}}>) {
// CHECK:             scf.condition(%[[VAL_29]]) %[[VAL_30]], %[[VAL_31]], %[[VAL_32]] : index, index, tensor<6x6xi32, #sparse{{[0-9]*}}>
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_33:.*]]: index, %[[VAL_34:.*]]: index, %[[VAL_35:.*]]: tensor<6x6xi32, #sparse{{[0-9]*}}>):
// CHECK:             %[[VAL_36:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_10]]] : memref<5xindex>
// CHECK:             %[[VAL_37:.*]]:4 = scf.for %[[VAL_38:.*]] = %[[VAL_10]] to %[[VAL_36]] step %[[VAL_7]] iter_args(%[[VAL_39:.*]] = %[[VAL_12]], %[[VAL_40:.*]] = %[[VAL_5]], %[[VAL_41:.*]] = %[[VAL_10]], %[[VAL_42:.*]] = %[[VAL_10]]) -> (i1, index, index, index) {
// CHECK:               %[[VAL_43:.*]] = arith.addi %[[VAL_38]], %[[VAL_9]] : index
// CHECK:               %[[VAL_44:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_43]]] : memref<5xindex>
// CHECK:               %[[VAL_45:.*]] = arith.addi %[[VAL_38]], %[[VAL_6]] : index
// CHECK:               %[[VAL_46:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_45]]] : memref<5xindex>
// CHECK:               %[[VAL_47:.*]] = arith.addi %[[VAL_38]], %[[VAL_4]] : index
// CHECK:               memref.store %[[VAL_42]], %[[VAL_20]]{{\[}}%[[VAL_47]]] : memref<5xindex>
// CHECK:               %[[VAL_48:.*]] = arith.addi %[[VAL_34]], %[[VAL_6]] : index
// CHECK:               %[[VAL_49:.*]]:5 = scf.while (%[[VAL_50:.*]] = %[[VAL_44]], %[[VAL_51:.*]] = %[[VAL_39]], %[[VAL_52:.*]] = %[[VAL_40]], %[[VAL_53:.*]] = %[[VAL_41]], %[[VAL_54:.*]] = %[[VAL_42]]) : (index, i1, index, index, index) -> (index, i1, index, index, index) {
// CHECK:                 %[[VAL_55:.*]] = arith.cmpi ult, %[[VAL_50]], %[[VAL_46]] : index
// CHECK:                 %[[VAL_56:.*]] = scf.if %[[VAL_55]] -> (i1) {
// CHECK:                   %[[VAL_57:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_50]]] : memref<?xindex>
// CHECK:                   %[[VAL_58:.*]] = arith.cmpi ult, %[[VAL_57]], %[[VAL_48]] : index
// CHECK:                   scf.yield %[[VAL_58]] : i1
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_12]] : i1
// CHECK:                 }
// CHECK:                 scf.condition(%[[VAL_56]]) %[[VAL_50]], %[[VAL_51]], %[[VAL_52]], %[[VAL_53]], %[[VAL_54]] : index, i1, index, index, index
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_59:.*]]: index, %[[VAL_60:.*]]: i1, %[[VAL_61:.*]]: index, %[[VAL_62:.*]]: index, %[[VAL_63:.*]]: index):
// CHECK:                 %[[VAL_64:.*]] = arith.addi %[[VAL_59]], %[[VAL_7]] : index
// CHECK:                 %[[VAL_65:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_59]]] : memref<?xindex>
// CHECK:                 %[[VAL_66:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_64]]] : memref<?xindex>
// CHECK:                 %[[VAL_67:.*]] = arith.cmpi ult, %[[VAL_65]], %[[VAL_66]] : index
// CHECK:                 %[[VAL_68:.*]] = arith.ori %[[VAL_67]], %[[VAL_60]] : i1
// CHECK:                 %[[VAL_69:.*]] = scf.if %[[VAL_67]] -> (index) {
// CHECK:                   %[[VAL_70:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_65]]] : memref<?xindex>
// CHECK:                   %[[VAL_71:.*]] = arith.cmpi ult, %[[VAL_70]], %[[VAL_61]] : index
// CHECK:                   %[[VAL_72:.*]] = arith.select %[[VAL_71]], %[[VAL_70]], %[[VAL_61]] : index
// CHECK:                   scf.yield %[[VAL_72]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_61]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_73:.*]] = arith.addi %[[VAL_62]], %[[VAL_9]] : index
// CHECK:                 memref.store %[[VAL_65]], %[[VAL_19]]{{\[}}%[[VAL_73]]] : memref<11xindex>
// CHECK:                 %[[VAL_74:.*]] = arith.addi %[[VAL_62]], %[[VAL_8]] : index
// CHECK:                 memref.store %[[VAL_66]], %[[VAL_19]]{{\[}}%[[VAL_74]]] : memref<11xindex>
// CHECK:                 %[[VAL_75:.*]] = arith.addi %[[VAL_62]], %[[VAL_7]] : index
// CHECK:                 %[[VAL_76:.*]] = arith.addi %[[VAL_63]], %[[VAL_7]] : index
// CHECK:                 scf.yield %[[VAL_64]], %[[VAL_68]], %[[VAL_69]], %[[VAL_75]], %[[VAL_76]] : index, i1, index, index, index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_77:.*]]#1, %[[VAL_77]]#2, %[[VAL_77]]#3, %[[VAL_77]]#4 : i1, index, index, index
// CHECK:             }
// CHECK:             memref.store %[[VAL_78:.*]]#2, %[[VAL_19]]{{\[}}%[[VAL_10]]] : memref<11xindex>
// CHECK:             memref.store %[[VAL_10]], %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<11xindex>
// CHECK:             %[[VAL_79:.*]] = arith.cmpi uge, %[[VAL_78]]#1, %[[VAL_6]] : index
// CHECK:             %[[VAL_80:.*]] = arith.andi %[[VAL_78]]#0, %[[VAL_79]] : i1
// CHECK:             %[[VAL_81:.*]] = arith.addi %[[VAL_78]]#1, %[[VAL_3]] : index
// CHECK:             %[[VAL_82:.*]] = arith.select %[[VAL_80]], %[[VAL_81]], %[[VAL_10]] : index
// CHECK:             %[[VAL_83:.*]]:3 = scf.while (%[[VAL_84:.*]] = %[[VAL_78]]#0, %[[VAL_85:.*]] = %[[VAL_78]]#1, %[[VAL_86:.*]] = %[[VAL_82]], %[[VAL_87:.*]] = %[[VAL_35]]) : (i1, index, index, tensor<6x6xi32, #sparse{{[0-9]*}}>) -> (index, index, tensor<6x6xi32, #sparse{{[0-9]*}}>) {
// CHECK:               scf.condition(%[[VAL_84]]) %[[VAL_85]], %[[VAL_86]], %[[VAL_87]] : index, index, tensor<6x6xi32, #sparse{{[0-9]*}}>
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_88:.*]]: index, %[[VAL_89:.*]]: index, %[[VAL_90:.*]]: tensor<6x6xi32, #sparse{{[0-9]*}}>):
// CHECK:               %[[VAL_91:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_7]]] : memref<5xindex>
// CHECK:               %[[VAL_92:.*]] = arith.addi %[[VAL_91]], %[[VAL_9]] : index
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_92]]] : memref<5xindex>
// CHECK:               %[[VAL_94:.*]] = arith.addi %[[VAL_91]], %[[VAL_6]] : index
// CHECK:               %[[VAL_95:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_94]]] : memref<5xindex>
// CHECK:               %[[VAL_96:.*]] = arith.addi %[[VAL_34]], %[[VAL_6]] : index
// CHECK:               %[[VAL_97:.*]]:3 = scf.while (%[[VAL_98:.*]] = %[[VAL_93]], %[[VAL_99:.*]] = %[[VAL_11]], %[[VAL_100:.*]] = %[[VAL_12]]) : (index, i32, i1) -> (index, i32, i1) {
// CHECK:                 %[[VAL_101:.*]] = arith.cmpi ult, %[[VAL_98]], %[[VAL_95]] : index
// CHECK:                 %[[VAL_102:.*]] = scf.if %[[VAL_101]] -> (i1) {
// CHECK:                   %[[VAL_103:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_98]]] : memref<?xindex>
// CHECK:                   %[[VAL_104:.*]] = arith.cmpi ult, %[[VAL_103]], %[[VAL_96]] : index
// CHECK:                   scf.yield %[[VAL_104]] : i1
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_12]] : i1
// CHECK:                 }
// CHECK:                 scf.condition(%[[VAL_102]]) %[[VAL_98]], %[[VAL_99]], %[[VAL_100]] : index, i32, i1
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_105:.*]]: index, %[[VAL_106:.*]]: i32, %[[VAL_107:.*]]: i1):
// CHECK:                 %[[VAL_108:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_105]]] : memref<?xindex>
// CHECK:                 %[[VAL_109:.*]] = arith.subi %[[VAL_108]], %[[VAL_34]] : index
// CHECK:                 %[[VAL_110:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<11xindex>
// CHECK:                 %[[VAL_111:.*]] = arith.addi %[[VAL_110]], %[[VAL_9]] : index
// CHECK:                 %[[VAL_112:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_111]]] : memref<11xindex>
// CHECK:                 %[[VAL_113:.*]] = arith.addi %[[VAL_110]], %[[VAL_8]] : index
// CHECK:                 %[[VAL_114:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_113]]] : memref<11xindex>
// CHECK:                 %[[VAL_115:.*]] = arith.addi %[[VAL_89]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_116:.*]]:2 = scf.while (%[[VAL_117:.*]] = %[[VAL_112]], %[[VAL_118:.*]] = %[[VAL_106]]) : (index, i32) -> (index, i32) {
// CHECK:                   %[[VAL_119:.*]] = arith.cmpi ult, %[[VAL_117]], %[[VAL_114]] : index
// CHECK:                   %[[VAL_120:.*]] = scf.if %[[VAL_119]] -> (i1) {
// CHECK:                     %[[VAL_121:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_117]]] : memref<?xindex>
// CHECK:                     %[[VAL_122:.*]] = arith.cmpi ult, %[[VAL_121]], %[[VAL_115]] : index
// CHECK:                     scf.yield %[[VAL_122]] : i1
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_12]] : i1
// CHECK:                   }
// CHECK:                   scf.condition(%[[VAL_120]]) %[[VAL_117]], %[[VAL_118]] : index, i32
// CHECK:                 } do {
// CHECK:                 ^bb0(%[[VAL_123:.*]]: index, %[[VAL_124:.*]]: i32):
// CHECK:                   %[[VAL_125:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_123]]] : memref<?xindex>
// CHECK:                   %[[VAL_126:.*]] = arith.subi %[[VAL_125]], %[[VAL_89]] : index
// CHECK:                   %[[VAL_127:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_123]]] : memref<?xi32>
// CHECK:                   %[[VAL_128:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_109]], %[[VAL_126]]] : tensor<3x3xi32>
// CHECK:                   %[[VAL_129:.*]] = arith.muli %[[VAL_127]], %[[VAL_128]] : i32
// CHECK:                   %[[VAL_130:.*]] = arith.addi %[[VAL_124]], %[[VAL_129]] : i32
// CHECK:                   %[[VAL_131:.*]] = arith.addi %[[VAL_123]], %[[VAL_7]] : index
// CHECK:                   scf.yield %[[VAL_131]], %[[VAL_130]] : index, i32
// CHECK:                 }
// CHECK:                 %[[VAL_132:.*]] = arith.addi %[[VAL_105]], %[[VAL_7]] : index
// CHECK:                 %[[VAL_133:.*]] = arith.addi %[[VAL_110]], %[[VAL_7]] : index
// CHECK:                 memref.store %[[VAL_133]], %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<11xindex>
// CHECK:                 scf.yield %[[VAL_132]], %[[VAL_134:.*]]#1, %[[VAL_2]] : index, i32, i1
// CHECK:               }
// CHECK:               %[[VAL_135:.*]] = scf.if %[[VAL_136:.*]]#2 -> (tensor<6x6xi32, #sparse{{[0-9]*}}>) {
// CHECK:                 %[[VAL_137:.*]] = sparse_tensor.insert %[[VAL_136]]#1 into %[[VAL_90]]{{\[}}%[[VAL_34]], %[[VAL_89]]] : tensor<6x6xi32, #sparse{{[0-9]*}}>
// CHECK:                 scf.yield %[[VAL_137]] : tensor<6x6xi32, #sparse{{[0-9]*}}>
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_90]] : tensor<6x6xi32, #sparse{{[0-9]*}}>
// CHECK:               }
// CHECK:               memref.store %[[VAL_10]], %[[VAL_20]]{{\[}}%[[VAL_7]]] : memref<5xindex>
// CHECK:               memref.store %[[VAL_10]], %[[VAL_19]]{{\[}}%[[VAL_7]]] : memref<11xindex>
// CHECK:               %[[VAL_138:.*]] = arith.cmpi ugt, %[[VAL_88]], %[[VAL_89]] : index
// CHECK:               %[[VAL_139:.*]]:3 = scf.if %[[VAL_138]] -> (index, i1, index) {
// CHECK:                 %[[VAL_140:.*]] = arith.addi %[[VAL_89]], %[[VAL_7]] : index
// CHECK:                 scf.yield %[[VAL_88]], %[[VAL_2]], %[[VAL_140]] : index, i1, index
// CHECK:               } else {
// CHECK:                 %[[VAL_141:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_10]]] : memref<11xindex>
// CHECK:                 %[[VAL_142:.*]]:2 = scf.for %[[VAL_143:.*]] = %[[VAL_10]] to %[[VAL_141]] step %[[VAL_7]] iter_args(%[[VAL_144:.*]] = %[[VAL_5]], %[[VAL_145:.*]] = %[[VAL_12]]) -> (index, i1) {
// CHECK:                   %[[VAL_146:.*]] = arith.addi %[[VAL_143]], %[[VAL_9]] : index
// CHECK:                   %[[VAL_147:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_146]]] : memref<11xindex>
// CHECK:                   %[[VAL_148:.*]] = arith.addi %[[VAL_143]], %[[VAL_8]] : index
// CHECK:                   %[[VAL_149:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_148]]] : memref<11xindex>
// CHECK:                   %[[VAL_150:.*]] = arith.cmpi ult, %[[VAL_147]], %[[VAL_149]] : index
// CHECK:                   %[[VAL_151:.*]] = scf.if %[[VAL_150]] -> (index) {
// CHECK:                     %[[VAL_152:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_147]]] : memref<?xindex>
// CHECK:                     %[[VAL_153:.*]] = arith.cmpi eq, %[[VAL_152]], %[[VAL_88]] : index
// CHECK:                     %[[VAL_154:.*]] = scf.if %[[VAL_153]] -> (index) {
// CHECK:                       %[[VAL_155:.*]] = arith.addi %[[VAL_147]], %[[VAL_7]] : index
// CHECK:                       memref.store %[[VAL_155]], %[[VAL_19]]{{\[}}%[[VAL_146]]] : memref<11xindex>
// CHECK:                       scf.yield %[[VAL_155]] : index
// CHECK:                     } else {
// CHECK:                       scf.yield %[[VAL_147]] : index
// CHECK:                     }
// CHECK:                     scf.yield %[[VAL_154]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_147]] : index
// CHECK:                   }
// CHECK:                   %[[VAL_156:.*]] = arith.cmpi ult, %[[VAL_151]], %[[VAL_149]] : index
// CHECK:                   %[[VAL_157:.*]] = scf.if %[[VAL_156]] -> (index) {
// CHECK:                     %[[VAL_158:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_151]]] : memref<?xindex>
// CHECK:                     scf.yield %[[VAL_158]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_144]] : index
// CHECK:                   }
// CHECK:                   %[[VAL_159:.*]] = arith.ori %[[VAL_156]], %[[VAL_145]] : i1
// CHECK:                   %[[VAL_160:.*]] = arith.cmpi ult, %[[VAL_157]], %[[VAL_144]] : index
// CHECK:                   %[[VAL_161:.*]] = arith.select %[[VAL_160]], %[[VAL_157]], %[[VAL_144]] : index
// CHECK:                   scf.yield %[[VAL_161]], %[[VAL_159]] : index, i1
// CHECK:                 }
// CHECK:                 %[[VAL_162:.*]] = arith.addi %[[VAL_163:.*]]#0, %[[VAL_7]] : index
// CHECK:                 %[[VAL_164:.*]] = arith.addi %[[VAL_163]]#0, %[[VAL_3]] : index
// CHECK:                 %[[VAL_165:.*]] = arith.cmpi uge, %[[VAL_162]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_166:.*]] = arith.select %[[VAL_165]], %[[VAL_164]], %[[VAL_10]] : index
// CHECK:                 scf.yield %[[VAL_163]]#0, %[[VAL_163]]#1, %[[VAL_166]] : index, i1, index
// CHECK:               }
// CHECK:               %[[VAL_167:.*]] = arith.addi %[[VAL_89]], %[[VAL_7]] : index
// CHECK:               %[[VAL_168:.*]] = arith.cmpi ugt, %[[VAL_169:.*]]#2, %[[VAL_167]] : index
// CHECK:               %[[VAL_170:.*]] = arith.select %[[VAL_168]], %[[VAL_169]]#2, %[[VAL_167]] : index
// CHECK:               %[[VAL_171:.*]] = arith.addi %[[VAL_170]], %[[VAL_6]] : index
// CHECK:               %[[VAL_172:.*]] = arith.cmpi ule, %[[VAL_171]], %[[VAL_5]] : index
// CHECK:               %[[VAL_173:.*]] = arith.andi %[[VAL_169]]#1, %[[VAL_172]] : i1
// CHECK:               scf.yield %[[VAL_173]], %[[VAL_169]]#0, %[[VAL_170]], %[[VAL_135]] : i1, index, index, tensor<6x6xi32, #sparse{{[0-9]*}}>
// CHECK:             }
// CHECK:             memref.store %[[VAL_10]], %[[VAL_20]]{{\[}}%[[VAL_7]]] : memref<5xindex>
// CHECK:             %[[VAL_174:.*]] = arith.cmpi ugt, %[[VAL_33]], %[[VAL_34]] : index
// CHECK:             %[[VAL_175:.*]]:3 = scf.if %[[VAL_174]] -> (index, i1, index) {
// CHECK:               %[[VAL_176:.*]] = arith.addi %[[VAL_34]], %[[VAL_7]] : index
// CHECK:               scf.yield %[[VAL_33]], %[[VAL_2]], %[[VAL_176]] : index, i1, index
// CHECK:             } else {
// CHECK:               %[[VAL_177:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_10]]] : memref<5xindex>
// CHECK:               %[[VAL_178:.*]]:2 = scf.for %[[VAL_179:.*]] = %[[VAL_10]] to %[[VAL_177]] step %[[VAL_7]] iter_args(%[[VAL_180:.*]] = %[[VAL_5]], %[[VAL_181:.*]] = %[[VAL_12]]) -> (index, i1) {
// CHECK:                 %[[VAL_182:.*]] = arith.addi %[[VAL_179]], %[[VAL_9]] : index
// CHECK:                 %[[VAL_183:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_182]]] : memref<5xindex>
// CHECK:                 %[[VAL_184:.*]] = arith.addi %[[VAL_179]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_185:.*]] = memref.load %[[VAL_20]]{{\[}}%[[VAL_184]]] : memref<5xindex>
// CHECK:                 %[[VAL_186:.*]] = arith.cmpi ult, %[[VAL_183]], %[[VAL_185]] : index
// CHECK:                 %[[VAL_187:.*]] = scf.if %[[VAL_186]] -> (index) {
// CHECK:                   %[[VAL_188:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_183]]] : memref<?xindex>
// CHECK:                   %[[VAL_189:.*]] = arith.cmpi eq, %[[VAL_188]], %[[VAL_33]] : index
// CHECK:                   %[[VAL_190:.*]] = scf.if %[[VAL_189]] -> (index) {
// CHECK:                     %[[VAL_191:.*]] = arith.addi %[[VAL_183]], %[[VAL_7]] : index
// CHECK:                     memref.store %[[VAL_191]], %[[VAL_20]]{{\[}}%[[VAL_182]]] : memref<5xindex>
// CHECK:                     scf.yield %[[VAL_191]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_183]] : index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_190]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_183]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_192:.*]] = arith.cmpi ult, %[[VAL_187]], %[[VAL_185]] : index
// CHECK:                 %[[VAL_193:.*]] = scf.if %[[VAL_192]] -> (index) {
// CHECK:                   %[[VAL_194:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_187]]] : memref<?xindex>
// CHECK:                   scf.yield %[[VAL_194]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_180]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_195:.*]] = arith.ori %[[VAL_192]], %[[VAL_181]] : i1
// CHECK:                 %[[VAL_196:.*]] = arith.cmpi ult, %[[VAL_193]], %[[VAL_180]] : index
// CHECK:                 %[[VAL_197:.*]] = arith.select %[[VAL_196]], %[[VAL_193]], %[[VAL_180]] : index
// CHECK:                 scf.yield %[[VAL_197]], %[[VAL_195]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_198:.*]] = arith.addi %[[VAL_199:.*]]#0, %[[VAL_7]] : index
// CHECK:               %[[VAL_200:.*]] = arith.addi %[[VAL_199]]#0, %[[VAL_3]] : index
// CHECK:               %[[VAL_201:.*]] = arith.cmpi uge, %[[VAL_198]], %[[VAL_6]] : index
// CHECK:               %[[VAL_202:.*]] = arith.select %[[VAL_201]], %[[VAL_200]], %[[VAL_10]] : index
// CHECK:               scf.yield %[[VAL_199]]#0, %[[VAL_199]]#1, %[[VAL_202]] : index, i1, index
// CHECK:             }
// CHECK:             %[[VAL_203:.*]] = arith.addi %[[VAL_34]], %[[VAL_7]] : index
// CHECK:             %[[VAL_204:.*]] = arith.cmpi ugt, %[[VAL_205:.*]]#2, %[[VAL_203]] : index
// CHECK:             %[[VAL_206:.*]] = arith.select %[[VAL_204]], %[[VAL_205]]#2, %[[VAL_203]] : index
// CHECK:             %[[VAL_207:.*]] = arith.addi %[[VAL_206]], %[[VAL_6]] : index
// CHECK:             %[[VAL_208:.*]] = arith.cmpi ule, %[[VAL_207]], %[[VAL_5]] : index
// CHECK:             %[[VAL_209:.*]] = arith.andi %[[VAL_205]]#1, %[[VAL_208]] : i1
// CHECK:             scf.yield %[[VAL_209]], %[[VAL_205]]#0, %[[VAL_206]], %[[VAL_210:.*]]#2 : i1, index, index, tensor<6x6xi32, #sparse{{[0-9]*}}>
// CHECK:           }
// CHECK:           %[[VAL_211:.*]] = sparse_tensor.load %[[VAL_212:.*]]#2 hasInserts : tensor<6x6xi32, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_211]] : tensor<6x6xi32, #sparse{{[0-9]*}}>
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
