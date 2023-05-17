// RUN: mlir-opt %s --sparsification="enable-index-reduction=true" --cse | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#DCSR = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

// CHECK-LABEL:   func.func @conv2d_all_sparse_CSR(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x8xi32, #{{.*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<3x3xi32>) -> tensor<6x6xi32, #{{.*}}> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_11:.*]] = bufferization.alloc_tensor() : tensor<6x6xi32, #{{.*}}>
// CHECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<8x8xi32, #{{.*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<8x8xi32, #{{.*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<8x8xi32, #{{.*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_15:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<8x8xi32, #{{.*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_16:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x8xi32, #{{.*}}> to memref<?xi32>
// CHECK-DAG:       %[[VAL_17:.*]] = bufferization.to_memref %[[VAL_1]] : memref<3x3xi32>
// CHECK-DAG:       %[[VAL_18:.*]] = memref.alloca(%[[VAL_2]]) : memref<?xindex>
// CHECK-DAG:       %[[VAL_19:.*]] = memref.alloca(%[[VAL_7]]) : memref<?xindex>
// CHECK:           %[[VAL_20:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:           memref.store %[[VAL_7]], %[[VAL_19]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_19]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:           memref.store %[[VAL_4]], %[[VAL_19]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// CHECK:           memref.store %[[VAL_20]], %[[VAL_19]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_21:.*]] = arith.cmpi ugt, %[[VAL_20]], %[[VAL_4]] : index
// CHECK:           %[[VAL_22:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:           %[[VAL_23:.*]] = arith.cmpi uge, %[[VAL_22]], %[[VAL_3]] : index
// CHECK:           %[[VAL_24:.*]] = arith.andi %[[VAL_21]], %[[VAL_23]] : i1
// CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_22]], %[[VAL_5]] : index
// CHECK:           %[[VAL_26:.*]] = arith.subi %[[VAL_25]], %[[VAL_3]] : index
// CHECK:           %[[VAL_27:.*]] = arith.select %[[VAL_24]], %[[VAL_26]], %[[VAL_4]] : index
// CHECK:           %[[VAL_28:.*]]:4 = scf.while (%[[VAL_29:.*]] = %[[VAL_21]], %[[VAL_30:.*]] = %[[VAL_22]], %[[VAL_31:.*]] = %[[VAL_27]], %[[VAL_32:.*]] = %[[VAL_11]]) : (i1, index, index, tensor<6x6xi32, #{{.*}}>) -> (i1, index, index, tensor<6x6xi32, #{{.*}}>) {
// CHECK:             scf.condition(%[[VAL_29]]) %[[VAL_29]], %[[VAL_30]], %[[VAL_31]], %[[VAL_32]] : i1, index, index, tensor<6x6xi32, #{{.*}}>
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_33:.*]]: i1, %[[VAL_34:.*]]: index, %[[VAL_35:.*]]: index, %[[VAL_36:.*]]: tensor<6x6xi32, #{{.*}}>):
// CHECK:             %[[VAL_37:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:             %[[VAL_38:.*]]:3 = scf.for %[[VAL_39:.*]] = %[[VAL_6]] to %[[VAL_37]] step %[[VAL_6]] iter_args(%[[VAL_40:.*]] = %[[VAL_10]], %[[VAL_41:.*]] = %[[VAL_2]], %[[VAL_42:.*]] = %[[VAL_6]]) -> (i1, index, index) {
// CHECK:               %[[VAL_43:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_39]]] : memref<?xindex>
// CHECK:               %[[VAL_44:.*]] = arith.addi %[[VAL_39]], %[[VAL_5]] : index
// CHECK:               %[[VAL_45:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_44]]] : memref<?xindex>
// CHECK:               %[[VAL_46:.*]] = arith.addi %[[VAL_35]], %[[VAL_3]] : index
// CHECK:               %[[VAL_47:.*]]:5 = scf.while (%[[VAL_48:.*]] = %[[VAL_43]], %[[VAL_49:.*]] = %[[VAL_9]], %[[VAL_50:.*]] = %[[VAL_40]], %[[VAL_51:.*]] = %[[VAL_41]], %[[VAL_52:.*]] = %[[VAL_42]]) : (index, i1, i1, index, index) -> (index, i1, i1, index, index) {
// CHECK:                 %[[VAL_53:.*]] = arith.cmpi ult, %[[VAL_48]], %[[VAL_45]] : index
// CHECK:                 %[[VAL_54:.*]] = arith.andi %[[VAL_49]], %[[VAL_53]] : i1
// CHECK:                 scf.condition(%[[VAL_54]]) %[[VAL_48]], %[[VAL_49]], %[[VAL_50]], %[[VAL_51]], %[[VAL_52]] : index, i1, i1, index, index
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_55:.*]]: index, %[[VAL_56:.*]]: i1, %[[VAL_57:.*]]: i1, %[[VAL_58:.*]]: index, %[[VAL_59:.*]]: index):
// CHECK:                 %[[VAL_60:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_55]]] : memref<?xindex>
// CHECK:                 %[[VAL_61:.*]] = arith.cmpi ult, %[[VAL_60]], %[[VAL_46]] : index
// CHECK:                 %[[VAL_62:.*]]:3 = scf.if %[[VAL_61]] -> (i1, index, index) {
// CHECK:                   %[[VAL_63:.*]] = arith.addi %[[VAL_55]], %[[VAL_5]] : index
// CHECK:                   %[[VAL_64:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_55]]] : memref<?xindex>
// CHECK:                   %[[VAL_65:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_63]]] : memref<?xindex>
// CHECK:                   %[[VAL_66:.*]] = arith.cmpi ult, %[[VAL_64]], %[[VAL_65]] : index
// CHECK:                   %[[VAL_67:.*]] = arith.ori %[[VAL_66]], %[[VAL_57]] : i1
// CHECK:                   %[[VAL_68:.*]] = scf.if %[[VAL_66]] -> (index) {
// CHECK:                     %[[VAL_69:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_64]]] : memref<?xindex>
// CHECK:                     %[[VAL_70:.*]] = arith.cmpi ult, %[[VAL_69]], %[[VAL_58]] : index
// CHECK:                     %[[VAL_71:.*]] = arith.select %[[VAL_70]], %[[VAL_69]], %[[VAL_58]] : index
// CHECK:                     scf.yield %[[VAL_71]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_58]] : index
// CHECK:                   }
// CHECK:                   memref.store %[[VAL_64]], %[[VAL_18]]{{\[}}%[[VAL_59]]] : memref<?xindex>
// CHECK:                   %[[VAL_72:.*]] = arith.addi %[[VAL_59]], %[[VAL_5]] : index
// CHECK:                   memref.store %[[VAL_65]], %[[VAL_18]]{{\[}}%[[VAL_72]]] : memref<?xindex>
// CHECK:                   %[[VAL_73:.*]] = arith.addi %[[VAL_59]], %[[VAL_6]] : index
// CHECK:                   scf.yield %[[VAL_67]], %[[VAL_74:.*]], %[[VAL_73]] : i1, index, index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_57]], %[[VAL_58]], %[[VAL_59]] : i1, index, index
// CHECK:                 } {"Emitted from" = "slice"}
// CHECK:                 %[[VAL_75:.*]] = arith.addi %[[VAL_55]], %[[VAL_5]] : index
// CHECK:                 scf.yield %[[VAL_75]], %[[VAL_61]], %[[VAL_76:.*]]#0, %[[VAL_76]]#1, %[[VAL_76]]#2 : index, i1, i1, index, index
// CHECK:               }
// CHECK:               scf.yield %[[VAL_77:.*]]#2, %[[VAL_77]]#3, %[[VAL_77]]#4 : i1, index, index
// CHECK:             }
// CHECK:             memref.store %[[VAL_78:.*]]#2, %[[VAL_18]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:             memref.store %[[VAL_4]], %[[VAL_18]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:             %[[VAL_79:.*]] = arith.cmpi uge, %[[VAL_78]]#1, %[[VAL_3]] : index
// CHECK:             %[[VAL_80:.*]] = arith.andi %[[VAL_78]]#0, %[[VAL_79]] : i1
// CHECK:             %[[VAL_81:.*]] = arith.addi %[[VAL_78]]#1, %[[VAL_5]] : index
// CHECK:             %[[VAL_82:.*]] = arith.subi %[[VAL_81]], %[[VAL_3]] : index
// CHECK:             %[[VAL_83:.*]] = arith.select %[[VAL_80]], %[[VAL_82]], %[[VAL_4]] : index
// CHECK:             %[[VAL_84:.*]]:4 = scf.while (%[[VAL_85:.*]] = %[[VAL_78]]#0, %[[VAL_86:.*]] = %[[VAL_78]]#1, %[[VAL_87:.*]] = %[[VAL_83]], %[[VAL_88:.*]] = %[[VAL_36]]) : (i1, index, index, tensor<6x6xi32, #{{.*}}>) -> (i1, index, index, tensor<6x6xi32, #{{.*}}>) {
// CHECK:               scf.condition(%[[VAL_85]]) %[[VAL_85]], %[[VAL_86]], %[[VAL_87]], %[[VAL_88]] : i1, index, index, tensor<6x6xi32, #{{.*}}>
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_89:.*]]: i1, %[[VAL_90:.*]]: index, %[[VAL_91:.*]]: index, %[[VAL_92:.*]]: tensor<6x6xi32, #{{.*}}>):
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:               %[[VAL_94:.*]] = arith.addi %[[VAL_93]], %[[VAL_6]] : index
// CHECK:               %[[VAL_95:.*]] = arith.addi %[[VAL_94]], %[[VAL_5]] : index
// CHECK:               %[[VAL_96:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_94]]] : memref<?xindex>
// CHECK:               %[[VAL_97:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_95]]] : memref<?xindex>
// CHECK:               %[[VAL_98:.*]] = arith.addi %[[VAL_35]], %[[VAL_3]] : index
// CHECK:               %[[VAL_99:.*]]:5 = scf.while (%[[VAL_100:.*]] = %[[VAL_96]], %[[VAL_101:.*]] = %[[VAL_9]], %[[VAL_102:.*]] = %[[VAL_8]], %[[VAL_103:.*]] = %[[VAL_10]], %[[VAL_104:.*]] = %[[VAL_92]]) : (index, i1, i32, i1, tensor<6x6xi32, #{{.*}}>) -> (index, i1, i32, i1, tensor<6x6xi32, #{{.*}}>) {
// CHECK:                 %[[VAL_105:.*]] = arith.cmpi ult, %[[VAL_100]], %[[VAL_97]] : index
// CHECK:                 %[[VAL_106:.*]] = arith.andi %[[VAL_101]], %[[VAL_105]] : i1
// CHECK:                 scf.condition(%[[VAL_106]]) %[[VAL_100]], %[[VAL_101]], %[[VAL_102]], %[[VAL_103]], %[[VAL_104]] : index, i1, i32, i1, tensor<6x6xi32, #{{.*}}>
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_107:.*]]: index, %[[VAL_108:.*]]: i1, %[[VAL_109:.*]]: i32, %[[VAL_110:.*]]: i1, %[[VAL_111:.*]]: tensor<6x6xi32, #{{.*}}>):
// CHECK:                 %[[VAL_112:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_107]]] : memref<?xindex>
// CHECK:                 %[[VAL_113:.*]] = arith.cmpi ult, %[[VAL_112]], %[[VAL_98]] : index
// CHECK:                 %[[VAL_114:.*]]:3 = scf.if %[[VAL_113]] -> (i32, i1, tensor<6x6xi32, #{{.*}}>) {
// CHECK:                   %[[VAL_115:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_107]]] : memref<?xindex>
// CHECK:                   %[[VAL_116:.*]] = arith.subi %[[VAL_115]], %[[VAL_35]] : index
// CHECK:                   %[[VAL_117:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:                   %[[VAL_118:.*]] = arith.addi %[[VAL_117]], %[[VAL_6]] : index
// CHECK:                   %[[VAL_119:.*]] = arith.addi %[[VAL_118]], %[[VAL_5]] : index
// CHECK:                   %[[VAL_120:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_118]]] : memref<?xindex>
// CHECK:                   %[[VAL_121:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_119]]] : memref<?xindex>
// CHECK:                   %[[VAL_122:.*]] = arith.addi %[[VAL_91]], %[[VAL_3]] : index
// CHECK:                   %[[VAL_123:.*]]:5 = scf.while (%[[VAL_124:.*]] = %[[VAL_120]], %[[VAL_125:.*]] = %[[VAL_9]], %[[VAL_126:.*]] = %[[VAL_109]], %[[VAL_127:.*]] = %[[VAL_110]], %[[VAL_128:.*]] = %[[VAL_111]]) : (index, i1, i32, i1, tensor<6x6xi32, #{{.*}}>) -> (index, i1, i32, i1, tensor<6x6xi32, #{{.*}}>) {
// CHECK:                     %[[VAL_129:.*]] = arith.cmpi ult, %[[VAL_124]], %[[VAL_121]] : index
// CHECK:                     %[[VAL_130:.*]] = arith.andi %[[VAL_125]], %[[VAL_129]] : i1
// CHECK:                     scf.condition(%[[VAL_130]]) %[[VAL_124]], %[[VAL_125]], %[[VAL_126]], %[[VAL_127]], %[[VAL_128]] : index, i1, i32, i1, tensor<6x6xi32, #{{.*}}>
// CHECK:                   } do {
// CHECK:                   ^bb0(%[[VAL_131:.*]]: index, %[[VAL_132:.*]]: i1, %[[VAL_133:.*]]: i32, %[[VAL_134:.*]]: i1, %[[VAL_135:.*]]: tensor<6x6xi32, #{{.*}}>):
// CHECK:                     %[[VAL_136:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_131]]] : memref<?xindex>
// CHECK:                     %[[VAL_137:.*]] = arith.cmpi ult, %[[VAL_136]], %[[VAL_122]] : index
// CHECK:                     %[[VAL_138:.*]]:3 = scf.if %[[VAL_137]] -> (i32, i1, tensor<6x6xi32, #{{.*}}>) {
// CHECK:                       %[[VAL_139:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_131]]] : memref<?xindex>
// CHECK:                       %[[VAL_140:.*]] = arith.subi %[[VAL_139]], %[[VAL_91]] : index
// CHECK:                       %[[VAL_141:.*]] = memref.load %[[VAL_16]]{{\[}}%[[VAL_131]]] : memref<?xi32>
// CHECK:                       %[[VAL_142:.*]] = memref.load %[[VAL_17]]{{\[}}%[[VAL_116]], %[[VAL_140]]] : memref<3x3xi32>
// CHECK:                       %[[VAL_143:.*]] = arith.muli %[[VAL_141]], %[[VAL_142]] : i32
// CHECK:                       %[[VAL_144:.*]] = arith.addi %[[VAL_133]], %[[VAL_143]] : i32
// CHECK:                       scf.yield %[[VAL_144]], %[[VAL_9]], %[[VAL_135]] : i32, i1, tensor<6x6xi32, #{{.*}}>
// CHECK:                     } else {
// CHECK:                       scf.yield %[[VAL_133]], %[[VAL_134]], %[[VAL_135]] : i32, i1, tensor<6x6xi32, #{{.*}}>
// CHECK:                     } {"Emitted from" = "slice"}
// CHECK:                     %[[VAL_145:.*]] = arith.addi %[[VAL_131]], %[[VAL_5]] : index
// CHECK:                     scf.yield %[[VAL_145]], %[[VAL_137]], %[[VAL_146:.*]]#0, %[[VAL_146]]#1, %[[VAL_146]]#2 : index, i1, i32, i1, tensor<6x6xi32, #{{.*}}>
// CHECK:                   } attributes {"Emitted from" = "linalg.generic"}
// CHECK:                   %[[VAL_147:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:                   %[[VAL_148:.*]] = arith.addi %[[VAL_147]], %[[VAL_6]] : index
// CHECK:                   memref.store %[[VAL_148]], %[[VAL_18]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:                   scf.yield %[[VAL_149:.*]]#2, %[[VAL_9]], %[[VAL_149]]#4 : i32, i1, tensor<6x6xi32, #{{.*}}>
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_109]], %[[VAL_110]], %[[VAL_111]] : i32, i1, tensor<6x6xi32, #{{.*}}>
// CHECK:                 } {"Emitted from" = "slice"}
// CHECK:                 %[[VAL_150:.*]] = arith.addi %[[VAL_107]], %[[VAL_5]] : index
// CHECK:                 scf.yield %[[VAL_150]], %[[VAL_113]], %[[VAL_151:.*]]#0, %[[VAL_151]]#1, %[[VAL_151]]#2 : index, i1, i32, i1, tensor<6x6xi32, #{{.*}}>
// CHECK:               } attributes {"Emitted from" = "linalg.generic"}
// CHECK:               %[[VAL_152:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:               %[[VAL_153:.*]] = arith.addi %[[VAL_152]], %[[VAL_6]] : index
// CHECK:               memref.store %[[VAL_153]], %[[VAL_19]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:               %[[VAL_154:.*]] = scf.if %[[VAL_155:.*]]#3 -> (tensor<6x6xi32, #{{.*}}>) {
// CHECK:                 %[[VAL_156:.*]] = sparse_tensor.insert %[[VAL_155]]#2 into %[[VAL_155]]#4{{\[}}%[[VAL_35]], %[[VAL_91]]] : tensor<6x6xi32, #{{.*}}>
// CHECK:                 scf.yield %[[VAL_156]] : tensor<6x6xi32, #{{.*}}>
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_157:.*]]#4 : tensor<6x6xi32, #{{.*}}>
// CHECK:               }
// CHECK:               memref.store %[[VAL_4]], %[[VAL_19]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:               memref.store %[[VAL_4]], %[[VAL_18]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:               %[[VAL_158:.*]] = arith.cmpi ugt, %[[VAL_90]], %[[VAL_91]] : index
// CHECK:               %[[VAL_159:.*]]:3 = scf.if %[[VAL_158]] -> (index, i1, index) {
// CHECK:                 %[[VAL_160:.*]] = arith.addi %[[VAL_91]], %[[VAL_5]] : index
// CHECK:                 scf.yield %[[VAL_90]], %[[VAL_89]], %[[VAL_160]] : index, i1, index
// CHECK:               } else {
// CHECK:                 %[[VAL_161:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:                 %[[VAL_162:.*]]:2 = scf.for %[[VAL_163:.*]] = %[[VAL_6]] to %[[VAL_161]] step %[[VAL_6]] iter_args(%[[VAL_164:.*]] = %[[VAL_2]], %[[VAL_165:.*]] = %[[VAL_10]]) -> (index, i1) {
// CHECK:                   %[[VAL_166:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_163]]] : memref<?xindex>
// CHECK:                   %[[VAL_167:.*]] = arith.addi %[[VAL_163]], %[[VAL_5]] : index
// CHECK:                   %[[VAL_168:.*]] = memref.load %[[VAL_18]]{{\[}}%[[VAL_167]]] : memref<?xindex>
// CHECK:                   %[[VAL_169:.*]] = arith.cmpi ult, %[[VAL_166]], %[[VAL_168]] : index
// CHECK:                   %[[VAL_170:.*]] = scf.if %[[VAL_169]] -> (index) {
// CHECK:                     %[[VAL_171:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_166]]] : memref<?xindex>
// CHECK:                     %[[VAL_172:.*]] = arith.cmpi eq, %[[VAL_171]], %[[VAL_90]] : index
// CHECK:                     %[[VAL_173:.*]] = scf.if %[[VAL_172]] -> (index) {
// CHECK:                       %[[VAL_174:.*]] = arith.addi %[[VAL_166]], %[[VAL_5]] : index
// CHECK:                       memref.store %[[VAL_174]], %[[VAL_18]]{{\[}}%[[VAL_163]]] : memref<?xindex>
// CHECK:                       scf.yield %[[VAL_174]] : index
// CHECK:                     } else {
// CHECK:                       scf.yield %[[VAL_166]] : index
// CHECK:                     }
// CHECK:                     scf.yield %[[VAL_175:.*]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_166]] : index
// CHECK:                   }
// CHECK:                   %[[VAL_176:.*]] = arith.cmpi ult, %[[VAL_177:.*]], %[[VAL_168]] : index
// CHECK:                   %[[VAL_178:.*]] = scf.if %[[VAL_176]] -> (index) {
// CHECK:                     %[[VAL_179:.*]] = memref.load %[[VAL_15]]{{\[}}%[[VAL_177]]] : memref<?xindex>
// CHECK:                     scf.yield %[[VAL_179]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_164]] : index
// CHECK:                   }
// CHECK:                   %[[VAL_180:.*]] = arith.ori %[[VAL_176]], %[[VAL_165]] : i1
// CHECK:                   %[[VAL_181:.*]] = arith.cmpi ult, %[[VAL_182:.*]], %[[VAL_164]] : index
// CHECK:                   %[[VAL_183:.*]] = arith.select %[[VAL_181]], %[[VAL_182]], %[[VAL_164]] : index
// CHECK:                   scf.yield %[[VAL_183]], %[[VAL_180]] : index, i1
// CHECK:                 }
// CHECK:                 %[[VAL_184:.*]] = arith.addi %[[VAL_185:.*]]#0, %[[VAL_5]] : index
// CHECK:                 %[[VAL_186:.*]] = arith.subi %[[VAL_184]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_187:.*]] = arith.cmpi uge, %[[VAL_184]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_188:.*]] = arith.select %[[VAL_187]], %[[VAL_186]], %[[VAL_4]] : index
// CHECK:                 scf.yield %[[VAL_185]]#0, %[[VAL_185]]#1, %[[VAL_188]] : index, i1, index
// CHECK:               }
// CHECK:               %[[VAL_189:.*]] = arith.addi %[[VAL_91]], %[[VAL_5]] : index
// CHECK:               %[[VAL_190:.*]] = arith.cmpi ugt, %[[VAL_191:.*]]#2, %[[VAL_189]] : index
// CHECK:               %[[VAL_192:.*]] = arith.select %[[VAL_190]], %[[VAL_191]]#2, %[[VAL_189]] : index
// CHECK:               %[[VAL_193:.*]] = arith.addi %[[VAL_192]], %[[VAL_3]] : index
// CHECK:               %[[VAL_194:.*]] = arith.cmpi ule, %[[VAL_193]], %[[VAL_2]] : index
// CHECK:               %[[VAL_195:.*]] = arith.andi %[[VAL_191]]#1, %[[VAL_194]] : i1
// CHECK:               scf.yield %[[VAL_195]], %[[VAL_191]]#0, %[[VAL_192]], %[[VAL_196:.*]] : i1, index, index, tensor<6x6xi32, #{{.*}}>
// CHECK:             } attributes {"Emitted from" = "linalg.generic"}
// CHECK:             memref.store %[[VAL_4]], %[[VAL_19]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:             %[[VAL_197:.*]] = arith.cmpi ugt, %[[VAL_34]], %[[VAL_35]] : index
// CHECK:             %[[VAL_198:.*]]:3 = scf.if %[[VAL_197]] -> (index, i1, index) {
// CHECK:               %[[VAL_199:.*]] = arith.addi %[[VAL_35]], %[[VAL_5]] : index
// CHECK:               scf.yield %[[VAL_34]], %[[VAL_33]], %[[VAL_199]] : index, i1, index
// CHECK:             } else {
// CHECK:               %[[VAL_200:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:               %[[VAL_201:.*]]:2 = scf.for %[[VAL_202:.*]] = %[[VAL_6]] to %[[VAL_200]] step %[[VAL_6]] iter_args(%[[VAL_203:.*]] = %[[VAL_2]], %[[VAL_204:.*]] = %[[VAL_10]]) -> (index, i1) {
// CHECK:                 %[[VAL_205:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_202]]] : memref<?xindex>
// CHECK:                 %[[VAL_206:.*]] = arith.addi %[[VAL_202]], %[[VAL_5]] : index
// CHECK:                 %[[VAL_207:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_206]]] : memref<?xindex>
// CHECK:                 %[[VAL_208:.*]] = arith.cmpi ult, %[[VAL_205]], %[[VAL_207]] : index
// CHECK:                 %[[VAL_209:.*]] = scf.if %[[VAL_208]] -> (index) {
// CHECK:                   %[[VAL_210:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_205]]] : memref<?xindex>
// CHECK:                   %[[VAL_211:.*]] = arith.cmpi eq, %[[VAL_210]], %[[VAL_34]] : index
// CHECK:                   %[[VAL_212:.*]] = scf.if %[[VAL_211]] -> (index) {
// CHECK:                     %[[VAL_213:.*]] = arith.addi %[[VAL_205]], %[[VAL_5]] : index
// CHECK:                     memref.store %[[VAL_213]], %[[VAL_19]]{{\[}}%[[VAL_202]]] : memref<?xindex>
// CHECK:                     scf.yield %[[VAL_213]] : index
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_205]] : index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_214:.*]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_205]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_215:.*]] = arith.cmpi ult, %[[VAL_216:.*]], %[[VAL_207]] : index
// CHECK:                 %[[VAL_217:.*]] = scf.if %[[VAL_215]] -> (index) {
// CHECK:                   %[[VAL_218:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_216]]] : memref<?xindex>
// CHECK:                   scf.yield %[[VAL_218]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_203]] : index
// CHECK:                 }
// CHECK:                 %[[VAL_219:.*]] = arith.ori %[[VAL_215]], %[[VAL_204]] : i1
// CHECK:                 %[[VAL_220:.*]] = arith.cmpi ult, %[[VAL_221:.*]], %[[VAL_203]] : index
// CHECK:                 %[[VAL_222:.*]] = arith.select %[[VAL_220]], %[[VAL_221]], %[[VAL_203]] : index
// CHECK:                 scf.yield %[[VAL_222]], %[[VAL_219]] : index, i1
// CHECK:               }
// CHECK:               %[[VAL_223:.*]] = arith.addi %[[VAL_224:.*]]#0, %[[VAL_5]] : index
// CHECK:               %[[VAL_225:.*]] = arith.subi %[[VAL_223]], %[[VAL_3]] : index
// CHECK:               %[[VAL_226:.*]] = arith.cmpi uge, %[[VAL_223]], %[[VAL_3]] : index
// CHECK:               %[[VAL_227:.*]] = arith.select %[[VAL_226]], %[[VAL_225]], %[[VAL_4]] : index
// CHECK:               scf.yield %[[VAL_224]]#0, %[[VAL_224]]#1, %[[VAL_227]] : index, i1, index
// CHECK:             }
// CHECK:             %[[VAL_228:.*]] = arith.addi %[[VAL_35]], %[[VAL_5]] : index
// CHECK:             %[[VAL_229:.*]] = arith.cmpi ugt, %[[VAL_230:.*]]#2, %[[VAL_228]] : index
// CHECK:             %[[VAL_231:.*]] = arith.select %[[VAL_229]], %[[VAL_230]]#2, %[[VAL_228]] : index
// CHECK:             %[[VAL_232:.*]] = arith.addi %[[VAL_231]], %[[VAL_3]] : index
// CHECK:             %[[VAL_233:.*]] = arith.cmpi ule, %[[VAL_232]], %[[VAL_2]] : index
// CHECK:             %[[VAL_234:.*]] = arith.andi %[[VAL_230]]#1, %[[VAL_233]] : i1
// CHECK:             scf.yield %[[VAL_234]], %[[VAL_230]]#0, %[[VAL_231]], %[[VAL_235:.*]]#3 : i1, index, index, tensor<6x6xi32, #{{.*}}>
// CHECK:           } attributes {"Emitted from" = "linalg.generic"}
// CHECK:           %[[VAL_236:.*]] = sparse_tensor.load %[[VAL_237:.*]]#3 hasInserts : tensor<6x6xi32, #{{.*}}>
// CHECK:           return %[[VAL_236]] : tensor<6x6xi32, #{{.*}}>
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
