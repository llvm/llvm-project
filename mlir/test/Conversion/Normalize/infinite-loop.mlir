// RUN: mlir-opt %s --normalize --mlir-use-nameloc-as-prefix 2>&1 | FileCheck %s

// CHECK-LABEL:   module {

// CHECK:           func.func @infinte_loop_v1(%[[ARG0:.*]]: memref<?xi32>, %[[ARG1:.*]]: i32) {
// CHECK:           %vl15969$e5677$ = arith.constant 1 : i32
// CHECK:           %vl15390$funcArg1-vl15969$ = arith.addi %[[ARG1]], %vl15969$e5677$ : i32
// CHECK:           cf.br ^bb1(%vl15390$funcArg1-vl15969$, %vl15390$funcArg1-vl15969$ : i32, i32)
// CHECK:           ^bb1(%0: i32, %1: i32):
// CHECK:           %vl85743$20b04$ = arith.constant 0 : i32
// CHECK:           %vl73800$blockArg0-vl85743$ = arith.muli %0, %vl85743$20b04$ : i32
// CHECK:           %vl85743$ded78$ = arith.constant -1 : i32
// CHECK:           %op51214$vl73800-vl85743$ = arith.xori %vl73800$blockArg0-vl85743$, %vl85743$ded78$ : i32
// CHECK:           %op12693$blockArg0-op51214$ = arith.addi %0, %op51214$vl73800-vl85743$ : i32
// CHECK:           %vl34407$blockArg1-vl85743$ = arith.addi %1, %vl85743$ded78$ : i32
// CHECK:           %op15672$op12693-vl34407$ = arith.addi %op12693$blockArg0-op51214$, %vl34407$blockArg1-vl85743$ : i32
// CHECK:           %op97825$op15672-vl73800$ = arith.muli %op15672$op12693-vl34407$, %vl73800$blockArg0-vl85743$ : i32
// CHECK:           %op51214$op97825-vl85743$ = arith.xori %op97825$op15672-vl73800$, %vl85743$ded78$ : i32
// CHECK:           %op12343$op15672-op51214$ = arith.addi %op15672$op12693-vl34407$, %op51214$op97825-vl85743$ : i32
// CHECK:           %op27844$op12343-vl85743$ = arith.addi %op12343$op15672-op51214$, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$ = arith.muli %op27844$op12343-vl85743$, %op97825$op15672-vl73800$ : i32
// CHECK:           %op51214$op97825-vl85743$_0 = arith.xori %op97825$op27844-op97825$, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$ = arith.addi %op27844$op12343-vl85743$, %op51214$op97825-vl85743$_0 : i32
// CHECK:           %op27844$op12343-vl85743$_1 = arith.addi %op12343$op27844-op51214$, %vl85743$20b04$ : i32
// CHECK:           %op27844$op27844-vl85743$ = arith.addi %op27844$op12343-vl85743$_1, %vl85743$20b04$ : i32
// CHECK:           %op27844$op27844-vl85743$_2 = arith.addi %op27844$op27844-vl85743$, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_3 = arith.muli %op27844$op12343-vl85743$_1, %op97825$op27844-op97825$ : i32
// CHECK:           %op97825$op27844-op97825$_4 = arith.muli %op27844$op27844-vl85743$_2, %op97825$op27844-op97825$_3 : i32
// CHECK:           %op51214$op97825-vl85743$_5 = arith.xori %op97825$op27844-op97825$_4, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_6 = arith.addi %op27844$op27844-vl85743$_2, %op51214$op97825-vl85743$_5 : i32
// CHECK:           %op27844$op12343-vl85743$_7 = arith.addi %op12343$op27844-op51214$_6, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_8 = arith.muli %op27844$op12343-vl85743$_7, %op97825$op27844-op97825$_4 : i32
// CHECK:           %op51214$op97825-vl85743$_9 = arith.xori %op97825$op27844-op97825$_8, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_10 = arith.addi %op27844$op12343-vl85743$_7, %op51214$op97825-vl85743$_9 : i32
// CHECK:           %op27844$op12343-vl85743$_11 = arith.addi %op12343$op27844-op51214$_10, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_12 = arith.muli %op27844$op12343-vl85743$_11, %op97825$op27844-op97825$_8 : i32
// CHECK:           %op51214$op97825-vl85743$_13 = arith.xori %op97825$op27844-op97825$_12, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_14 = arith.addi %op27844$op12343-vl85743$_11, %op51214$op97825-vl85743$_13 : i32
// CHECK:           %op27844$op12343-vl85743$_15 = arith.addi %op12343$op27844-op51214$_14, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_16 = arith.muli %op27844$op12343-vl85743$_15, %op97825$op27844-op97825$_12 : i32
// CHECK:           %op51214$op97825-vl85743$_17 = arith.xori %op97825$op27844-op97825$_16, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_18 = arith.addi %op27844$op12343-vl85743$_15, %op51214$op97825-vl85743$_17 : i32
// CHECK:           %op27844$op12343-vl85743$_19 = arith.addi %op12343$op27844-op51214$_18, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_20 = arith.muli %op27844$op12343-vl85743$_19, %op97825$op27844-op97825$_16 : i32
// CHECK:           %op51214$op97825-vl85743$_21 = arith.xori %op97825$op27844-op97825$_20, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_22 = arith.addi %op27844$op12343-vl85743$_19, %op51214$op97825-vl85743$_21 : i32
// CHECK:           %vl85743$51850$ = arith.constant -9 : i32
// CHECK:           %vl34407$blockArg1-vl85743$_23 = arith.addi %1, %vl85743$51850$ : i32
// CHECK:           %op17008$vl34407-vl85743$ = arith.muli %vl34407$blockArg1-vl85743$_23, %vl85743$20b04$ : i32
// CHECK:           %op51214$op17008-vl85743$ = arith.xori %op17008$vl34407-vl85743$, %vl85743$ded78$ : i32
// CHECK:           %op12343$op51214-vl34407$ = arith.addi %op51214$op17008-vl85743$, %vl34407$blockArg1-vl85743$_23 : i32
// CHECK:           %op15672$op12343-op12343$ = arith.addi %op12343$op27844-op51214$_22, %op12343$op51214-vl34407$ : i32
// CHECK:           %op97825$op15672-op97825$ = arith.muli %op15672$op12343-op12343$, %op97825$op27844-op97825$_20 : i32
// CHECK:           %op51214$op97825-vl85743$_24 = arith.xori %op97825$op15672-op97825$, %vl85743$ded78$ : i32
// CHECK:           %op12343$op15672-op51214$_25 = arith.addi %op15672$op12343-op12343$, %op51214$op97825-vl85743$_24 : i32
// CHECK:           %op17008$vl34407-vl85743$_26 = arith.muli %vl34407$blockArg1-vl85743$, %vl85743$20b04$ : i32
// CHECK:           %op51214$op17008-vl85743$_27 = arith.xori %op17008$vl34407-vl85743$_26, %vl85743$ded78$ : i32
// CHECK:           %op12343$op51214-vl34407$_28 = arith.addi %op51214$op17008-vl85743$_27, %vl34407$blockArg1-vl85743$ : i32
// CHECK:           %op97825$op12343-op17008$ = arith.muli %op12343$op51214-vl34407$_28, %op17008$vl34407-vl85743$_26 : i32
// CHECK:           %op51214$op97825-vl85743$_29 = arith.xori %op97825$op12343-op17008$, %vl85743$ded78$ : i32
// CHECK:           %op12343$op12343-op51214$ = arith.addi %op12343$op51214-vl34407$_28, %op51214$op97825-vl85743$_29 : i32
// CHECK:           %op27844$op12343-vl85743$_30 = arith.addi %op12343$op12343-op51214$, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_31 = arith.muli %op27844$op12343-vl85743$_30, %op97825$op12343-op17008$ : i32
// CHECK:           %op51214$op97825-vl85743$_32 = arith.xori %op97825$op27844-op97825$_31, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_33 = arith.addi %op27844$op12343-vl85743$_30, %op51214$op97825-vl85743$_32 : i32
// CHECK:           %op15672$op12343-op12343$_34 = arith.addi %op12343$op15672-op51214$_25, %op12343$op27844-op51214$_33 : i32
// CHECK:           %op97825$op15672-op97825$_35 = arith.muli %op15672$op12343-op12343$_34, %op97825$op15672-op97825$ : i32
// CHECK:           %op51214$op97825-vl85743$_36 = arith.xori %op97825$op15672-op97825$_35, %vl85743$ded78$ : i32
// CHECK:           %op12343$op15672-op51214$_37 = arith.addi %op15672$op12343-op12343$_34, %op51214$op97825-vl85743$_36 : i32
// CHECK:           %op27844$op12343-vl85743$_38 = arith.addi %op12343$op15672-op51214$_37, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_39 = arith.muli %op27844$op12343-vl85743$_38, %op97825$op15672-op97825$_35 : i32
// CHECK:           %op51214$op97825-vl85743$_40 = arith.xori %op97825$op27844-op97825$_39, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_41 = arith.addi %op27844$op12343-vl85743$_38, %op51214$op97825-vl85743$_40 : i32
// CHECK:           %op27844$op12343-vl85743$_42 = arith.addi %op12343$op27844-op51214$_41, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_43 = arith.muli %op27844$op12343-vl85743$_42, %op97825$op27844-op97825$_39 : i32
// CHECK:           %op51214$op97825-vl85743$_44 = arith.xori %op97825$op27844-op97825$_43, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_45 = arith.addi %op27844$op12343-vl85743$_42, %op51214$op97825-vl85743$_44 : i32
// CHECK:           %vl85743$7b7de$ = arith.constant -14 : i32
// CHECK:           %vl34407$blockArg1-vl85743$_46 = arith.addi %1, %vl85743$7b7de$ : i32
// CHECK:           %op17008$vl34407-vl85743$_47 = arith.muli %vl34407$blockArg1-vl85743$_46, %vl85743$20b04$ : i32
// CHECK:           %op51214$op17008-vl85743$_48 = arith.xori %op17008$vl34407-vl85743$_47, %vl85743$ded78$ : i32
// CHECK:           %op12343$op51214-vl34407$_49 = arith.addi %op51214$op17008-vl85743$_48, %vl34407$blockArg1-vl85743$_46 : i32
// CHECK:           %op97825$op12343-op17008$_50 = arith.muli %op12343$op51214-vl34407$_49, %op17008$vl34407-vl85743$_47 : i32
// CHECK:           %op51214$op97825-vl85743$_51 = arith.xori %op97825$op12343-op17008$_50, %vl85743$ded78$ : i32
// CHECK:           %op12343$op12343-op51214$_52 = arith.addi %op12343$op51214-vl34407$_49, %op51214$op97825-vl85743$_51 : i32
// CHECK:           %op27844$op12343-vl85743$_53 = arith.addi %op12343$op12343-op51214$_52, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_54 = arith.muli %op27844$op12343-vl85743$_53, %op97825$op12343-op17008$_50 : i32
// CHECK:           %op51214$op97825-vl85743$_55 = arith.xori %op97825$op27844-op97825$_54, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_56 = arith.addi %op27844$op12343-vl85743$_53, %op51214$op97825-vl85743$_55 : i32
// CHECK:           %op27844$op12343-vl85743$_57 = arith.addi %op12343$op27844-op51214$_56, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_58 = arith.muli %op27844$op12343-vl85743$_57, %op97825$op27844-op97825$_54 : i32
// CHECK:           %op51214$op97825-vl85743$_59 = arith.xori %op97825$op27844-op97825$_58, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_60 = arith.addi %op27844$op12343-vl85743$_57, %op51214$op97825-vl85743$_59 : i32
// CHECK:           %op27844$op12343-vl85743$_61 = arith.addi %op12343$op27844-op51214$_60, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_62 = arith.muli %op27844$op12343-vl85743$_61, %op97825$op27844-op97825$_58 : i32
// CHECK:           %op51214$op97825-vl85743$_63 = arith.xori %op97825$op27844-op97825$_62, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_64 = arith.addi %op27844$op12343-vl85743$_61, %op51214$op97825-vl85743$_63 : i32
// CHECK:           %op27844$op12343-vl85743$_65 = arith.addi %op12343$op27844-op51214$_64, %vl85743$20b04$ : i32
// CHECK:           %op15672$op12343-op27844$ = arith.addi %op12343$op27844-op51214$_45, %op27844$op12343-vl85743$_65 : i32
// CHECK:           %op97825$op15672-op97825$_66 = arith.muli %op15672$op12343-op27844$, %op97825$op27844-op97825$_43 : i32
// CHECK:           %op51214$op97825-vl85743$_67 = arith.xori %op97825$op15672-op97825$_66, %vl85743$ded78$ : i32
// CHECK:           %op12343$op15672-op51214$_68 = arith.addi %op15672$op12343-op27844$, %op51214$op97825-vl85743$_67 : i32
// CHECK:           %op15672$op12343-vl34407$ = arith.addi %op12343$op15672-op51214$_68, %vl34407$blockArg1-vl85743$_46 : i32
// CHECK:           %op97825$op15672-op97825$_69 = arith.muli %op15672$op12343-vl34407$, %op97825$op15672-op97825$_66 : i32
// CHECK:           %op51214$op97825-vl85743$_70 = arith.xori %op97825$op15672-op97825$_69, %vl85743$ded78$ : i32
// CHECK:           %op12343$op15672-op51214$_71 = arith.addi %op15672$op12343-vl34407$, %op51214$op97825-vl85743$_70 : i32
// CHECK:           %vl85743$1e72e$ = arith.constant -21 : i32
// CHECK:           %vl34407$blockArg1-vl85743$_72 = arith.addi %1, %vl85743$1e72e$ : i32
// CHECK:           %op17008$vl34407-vl85743$_73 = arith.muli %vl34407$blockArg1-vl85743$_72, %vl85743$20b04$ : i32
// CHECK:           %op51214$op17008-vl85743$_74 = arith.xori %op17008$vl34407-vl85743$_73, %vl85743$ded78$ : i32
// CHECK:           %op12343$op51214-vl34407$_75 = arith.addi %op51214$op17008-vl85743$_74, %vl34407$blockArg1-vl85743$_72 : i32
// CHECK:           %op97825$op12343-op17008$_76 = arith.muli %op12343$op51214-vl34407$_75, %op17008$vl34407-vl85743$_73 : i32
// CHECK:           %op51214$op97825-vl85743$_77 = arith.xori %op97825$op12343-op17008$_76, %vl85743$ded78$ : i32
// CHECK:           %op12343$op12343-op51214$_78 = arith.addi %op12343$op51214-vl34407$_75, %op51214$op97825-vl85743$_77 : i32
// CHECK:           %op27844$op12343-vl85743$_79 = arith.addi %op12343$op12343-op51214$_78, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_80 = arith.muli %op27844$op12343-vl85743$_79, %op97825$op12343-op17008$_76 : i32
// CHECK:           %op51214$op97825-vl85743$_81 = arith.xori %op97825$op27844-op97825$_80, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_82 = arith.addi %op27844$op12343-vl85743$_79, %op51214$op97825-vl85743$_81 : i32
// CHECK:           %op27844$op12343-vl85743$_83 = arith.addi %op12343$op27844-op51214$_82, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_84 = arith.muli %op27844$op12343-vl85743$_83, %op97825$op27844-op97825$_80 : i32
// CHECK:           %op51214$op97825-vl85743$_85 = arith.xori %op97825$op27844-op97825$_84, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_86 = arith.addi %op27844$op12343-vl85743$_83, %op51214$op97825-vl85743$_85 : i32
// CHECK:           %op27844$op12343-vl85743$_87 = arith.addi %op12343$op27844-op51214$_86, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_88 = arith.muli %op27844$op12343-vl85743$_87, %op97825$op27844-op97825$_84 : i32
// CHECK:           %op51214$op97825-vl85743$_89 = arith.xori %op97825$op27844-op97825$_88, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_90 = arith.addi %op27844$op12343-vl85743$_87, %op51214$op97825-vl85743$_89 : i32
// CHECK:           %op27844$op12343-vl85743$_91 = arith.addi %op12343$op27844-op51214$_90, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_92 = arith.muli %op27844$op12343-vl85743$_91, %op97825$op27844-op97825$_88 : i32
// CHECK:           %op13782$op12343-op97825$ = arith.addi %op12343$op15672-op51214$_71, %op97825$op27844-op97825$_92 : i32
// CHECK:           %op97825$op13782-op97825$ = arith.muli %op13782$op12343-op97825$, %op97825$op15672-op97825$_69 : i32
// CHECK:           %op51214$op97825-vl85743$_93 = arith.xori %op97825$op13782-op97825$, %vl85743$ded78$ : i32
// CHECK:           %op12343$op13782-op51214$ = arith.addi %op13782$op12343-op97825$, %op51214$op97825-vl85743$_93 : i32
// CHECK:           %op27844$op12343-vl85743$_94 = arith.addi %op12343$op13782-op51214$, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_95 = arith.muli %op27844$op12343-vl85743$_94, %op97825$op13782-op97825$ : i32
// CHECK:           %op51214$op97825-vl85743$_96 = arith.xori %op97825$op27844-op97825$_95, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_97 = arith.addi %op27844$op12343-vl85743$_94, %op51214$op97825-vl85743$_96 : i32
// CHECK:           %op27844$op12343-vl85743$_98 = arith.addi %op12343$op27844-op51214$_97, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_99 = arith.muli %op27844$op12343-vl85743$_98, %op97825$op27844-op97825$_95 : i32
// CHECK:           %op51214$op97825-vl85743$_100 = arith.xori %op97825$op27844-op97825$_99, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_101 = arith.addi %op27844$op12343-vl85743$_98, %op51214$op97825-vl85743$_100 : i32
// CHECK:           %op27844$op12343-vl85743$_102 = arith.addi %op12343$op27844-op51214$_101, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_103 = arith.muli %op27844$op12343-vl85743$_102, %op97825$op27844-op97825$_99 : i32
// CHECK:           %op51214$op97825-vl85743$_104 = arith.xori %op97825$op27844-op97825$_103, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_105 = arith.addi %op27844$op12343-vl85743$_102, %op51214$op97825-vl85743$_104 : i32
// CHECK:           %op27844$op12343-vl85743$_106 = arith.addi %op12343$op27844-op51214$_105, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_107 = arith.muli %op27844$op12343-vl85743$_106, %op97825$op27844-op97825$_103 : i32
// CHECK:           %op51214$op97825-vl85743$_108 = arith.xori %op97825$op27844-op97825$_107, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_109 = arith.addi %op27844$op12343-vl85743$_106, %op51214$op97825-vl85743$_108 : i32
// CHECK:           %op27844$op12343-vl85743$_110 = arith.addi %op12343$op27844-op51214$_109, %vl85743$20b04$ : i32
// CHECK:           %op27844$op27844-vl85743$_111 = arith.addi %op27844$op12343-vl85743$_110, %vl85743$20b04$ : i32
// CHECK:           %op15672$op27844-vl34407$ = arith.addi %op27844$op27844-vl85743$_111, %vl34407$blockArg1-vl85743$_72 : i32
// CHECK:           %vl16744$e527e$ = arith.constant 0 : index
// CHECK:           memref.store %op15672$op27844-vl34407$, %[[ARG0]][%vl16744$e527e$] : memref<?xi32>
// CHECK:           cf.br ^bb1(%op15672$op27844-vl34407$, %vl34407$blockArg1-vl85743$_72 : i32, i32)
// CHECK:         }
func.func @infinte_loop_v1(%arg0: memref<?xi32>, %arg1: i32) {
    %c1 = arith.constant 1 : i32
    %a = arith.addi %arg1, %c1 : i32
    cf.br ^bb1(%a, %a : i32, i32)

    ^bb1(%tmp: i32, %tmp2: i32):
    %c0 = arith.constant 0 : i32
    %cneg1 = arith.constant -1 : i32
    %cneg9 = arith.constant -9 : i32
    %cneg14 = arith.constant -14 : i32
    %cneg21 = arith.constant -21 : i32
    
    // Branch E: Start with tmp2-based constants
    %e1 = arith.addi %tmp2, %cneg14 : i32
    
    // Branch A: First main chain
    %tmp3 = arith.muli %tmp, %c0 : i32
    %tmp4 = arith.xori %tmp3, %cneg1 : i32
    
    // Branch D: Another tmp2-based chain
    %d1 = arith.addi %tmp2, %cneg21 : i32
    %d2 = arith.muli %d1, %c0 : i32
    
    // Branch B: tmp2-cneg1 chain
    %tmp6 = arith.addi %tmp2, %cneg1 : i32
    
    // Branch A continued
    %tmp5 = arith.addi %tmp, %tmp4 : i32
    
    // Branch C: tmp2-cneg9 chain  
    %base1 = arith.addi %tmp2, %cneg9 : i32
    
    // Branch E continued
    %e2 = arith.muli %e1, %c0 : i32
    %e3 = arith.xori %e2, %cneg1 : i32
    
    // Branch A continued
    %tmp7 = arith.addi %tmp5, %tmp6 : i32
    
    // Branch D continued
    %d3 = arith.xori %d2, %cneg1 : i32
    
    // Branch B continued
    %alt1 = arith.muli %tmp6, %c0 : i32
    %alt2 = arith.xori %alt1, %cneg1 : i32
    
    // Branch C continued
    %base2 = arith.muli %base1, %c0 : i32
    
    // Branch A continued
    %tmp8 = arith.muli %tmp7, %tmp3 : i32
    %tmp9 = arith.xori %tmp8, %cneg1 : i32
    
    // Branch E continued
    %e4 = arith.addi %e1, %e3 : i32
    
    // Branch B continued
    %alt3 = arith.addi %tmp6, %alt2 : i32
    
    // Branch D continued
    %d4 = arith.addi %d1, %d3 : i32
    %d5 = arith.muli %d4, %d2 : i32
    
    // Branch A continued
    %tmp10 = arith.addi %tmp7, %tmp9 : i32
    
    // Branch C continued
    %base3 = arith.xori %base2, %cneg1 : i32
    %base4 = arith.addi %base1, %base3 : i32
    
    // Branch B continued
    %alt4 = arith.muli %alt3, %alt1 : i32
    %alt5 = arith.xori %alt4, %cneg1 : i32
    
    // Branch A continued
    %tmp11 = arith.addi %tmp10, %c0 : i32
    %tmp12 = arith.muli %tmp11, %tmp8 : i32
    
    // Branch E continued
    %e5 = arith.muli %e4, %e2 : i32
    %e6 = arith.xori %e5, %cneg1 : i32
    
    // Branch D continued
    %d6 = arith.xori %d5, %cneg1 : i32
    
    // Branch A continued
    %tmp13 = arith.xori %tmp12, %cneg1 : i32
    %tmp14 = arith.addi %tmp11, %tmp13 : i32
    
    // Branch B continued
    %alt6 = arith.addi %alt3, %alt5 : i32
    %alt7 = arith.addi %alt6, %c0 : i32
    
    // Branch E continued
    %e7 = arith.addi %e4, %e6 : i32
    
    // Branch A continued
    %tmp15 = arith.addi %tmp14, %c0 : i32
    %tmp16 = arith.muli %tmp15, %tmp12 : i32
    
    // Branch D continued
    %d7 = arith.addi %d4, %d6 : i32
    %d8 = arith.addi %d7, %c0 : i32
    
    // Branch B continued
    %alt8 = arith.muli %alt7, %alt4 : i32
    
    // Branch A continued
    %tmp17 = arith.addi %tmp15, %c0 : i32
    %tmp18 = arith.addi %tmp17, %c0 : i32
    
    // Branch E continued
    %e8 = arith.addi %e7, %c0 : i32
    %e9 = arith.muli %e8, %e5 : i32
    
    // Branch A continued
    %tmp19 = arith.muli %tmp18, %tmp16 : i32
    %tmp20 = arith.xori %tmp19, %cneg1 : i32
    
    // Branch D continued
    %d9 = arith.muli %d8, %d5 : i32
    
    // Branch B continued
    %alt9 = arith.xori %alt8, %cneg1 : i32
    %alt10 = arith.addi %alt7, %alt9 : i32
    
    // Branch A continued
    %tmp21 = arith.addi %tmp18, %tmp20 : i32
    
    // Branch E continued
    %e10 = arith.xori %e9, %cneg1 : i32
    
    // Branch A continued
    %tmp22 = arith.addi %tmp21, %c0 : i32
    %tmp23 = arith.muli %tmp22, %tmp19 : i32
    
    // Branch D continued
    %d10 = arith.xori %d9, %cneg1 : i32
    %d11 = arith.addi %d8, %d10 : i32
    
    // Branch E continued
    %e11 = arith.addi %e8, %e10 : i32
    
    // Branch A continued
    %tmp24 = arith.xori %tmp23, %cneg1 : i32
    %tmp25 = arith.addi %tmp22, %tmp24 : i32
    
    // Branch D continued
    %d12 = arith.addi %d11, %c0 : i32
    
    // Branch A continued
    %tmp26 = arith.addi %tmp25, %c0 : i32
    %tmp27 = arith.muli %tmp26, %tmp23 : i32
    
    // Branch E continued
    %e12 = arith.addi %e11, %c0 : i32
    
    // Branch A continued
    %tmp28 = arith.xori %tmp27, %cneg1 : i32
    %tmp29 = arith.addi %tmp26, %tmp28 : i32
    
    // Branch D continued
    %d13 = arith.muli %d12, %d9 : i32
    %d14 = arith.xori %d13, %cneg1 : i32
    
    // Branch A continued
    %tmp30 = arith.addi %tmp29, %c0 : i32
    %tmp31 = arith.muli %tmp30, %tmp27 : i32
    
    // Branch E continued
    %e13 = arith.muli %e12, %e9 : i32
    
    // Branch A continued
    %tmp32 = arith.xori %tmp31, %cneg1 : i32
    %tmp33 = arith.addi %tmp30, %tmp32 : i32
    
    // Branch D continued
    %d15 = arith.addi %d12, %d14 : i32
    
    // Branch A continued
    %tmp34 = arith.addi %tmp33, %c0 : i32
    %tmp35 = arith.muli %tmp34, %tmp31 : i32
    
    // Branch E continued
    %e14 = arith.xori %e13, %cneg1 : i32
    %e15 = arith.addi %e12, %e14 : i32
    
    // Branch A continued
    %tmp36 = arith.xori %tmp35, %cneg1 : i32
    %tmp37 = arith.addi %tmp34, %tmp36 : i32
    
    // Branch D continued
    %d16 = arith.addi %d15, %c0 : i32
    
    // First merge: A + C
    %tmp39 = arith.addi %tmp37, %base4 : i32
    
    // Branch E continued
    %e16 = arith.addi %e15, %c0 : i32
    
    // Continue merged chain
    %tmp40 = arith.muli %tmp39, %tmp35 : i32
    
    // Branch D continued
    %d17 = arith.muli %d16, %d13 : i32
    
    // Continue merged chain
    %tmp41 = arith.xori %tmp40, %cneg1 : i32
    %tmp42 = arith.addi %tmp39, %tmp41 : i32
    
    // Branch E continued
    %e17 = arith.muli %e16, %e13 : i32
    %e18 = arith.xori %e17, %cneg1 : i32
    
    // Second merge: AB + B
    %tmp43 = arith.addi %tmp42, %alt10 : i32
    
    // Branch D continued
    %d18 = arith.xori %d17, %cneg1 : i32
    %d19 = arith.addi %d16, %d18 : i32
    
    // Continue merged chain
    %tmp44 = arith.muli %tmp43, %tmp40 : i32
    
    // Branch E continued
    %e19 = arith.addi %e16, %e18 : i32
    
    // Continue merged chain
    %tmp45 = arith.xori %tmp44, %cneg1 : i32
    %tmp46 = arith.addi %tmp43, %tmp45 : i32
    
    // Branch D continued
    %d20 = arith.addi %d19, %c0 : i32
    
    // Continue merged chain
    %tmp47 = arith.addi %tmp46, %c0 : i32
    
    // Branch E final
    %e20 = arith.addi %e19, %c0 : i32
    
    // Continue merged chain
    %tmp48 = arith.muli %tmp47, %tmp44 : i32
    %tmp49 = arith.xori %tmp48, %cneg1 : i32
    
    // Branch D final
    %d21 = arith.muli %d20, %d17 : i32
    
    // Continue merged chain
    %tmp50 = arith.addi %tmp47, %tmp49 : i32
    %tmp51 = arith.addi %tmp50, %c0 : i32
    
    // Merge in E
    %tmp52 = arith.muli %tmp51, %tmp48 : i32
    %tmp53 = arith.xori %tmp52, %cneg1 : i32
    %tmp54 = arith.addi %tmp51, %tmp53 : i32
    
    // Continue with E merge
    %tmp55 = arith.addi %tmp54, %e20 : i32
    
    // Continue chain
    %tmp56 = arith.muli %tmp55, %tmp52 : i32
    %tmp57 = arith.xori %tmp56, %cneg1 : i32
    %tmp58 = arith.addi %tmp55, %tmp57 : i32
    
    // Merge in E branch value
    %tmp60 = arith.addi %tmp58, %e1 : i32
    
    // Continue chain
    %tmp61 = arith.muli %tmp60, %tmp56 : i32
    %tmp62 = arith.xori %tmp61, %cneg1 : i32
    %tmp63 = arith.addi %tmp60, %tmp62 : i32
    
    // Merge in D
    %tmp64 = arith.addi %tmp63, %d21 : i32
    
    // Continue chain
    %tmp65 = arith.muli %tmp64, %tmp61 : i32
    %tmp66 = arith.xori %tmp65, %cneg1 : i32
    %tmp67 = arith.addi %tmp64, %tmp66 : i32
    %tmp68 = arith.addi %tmp67, %c0 : i32
    %tmp69 = arith.muli %tmp68, %tmp65 : i32
    %tmp70 = arith.xori %tmp69, %cneg1 : i32
    %tmp71 = arith.addi %tmp68, %tmp70 : i32
    %tmp72 = arith.addi %tmp71, %c0 : i32
    %tmp73 = arith.muli %tmp72, %tmp69 : i32
    %tmp74 = arith.xori %tmp73, %cneg1 : i32
    %tmp75 = arith.addi %tmp72, %tmp74 : i32
    %tmp76 = arith.addi %tmp75, %c0 : i32
    %tmp77 = arith.muli %tmp76, %tmp73 : i32
    %tmp78 = arith.xori %tmp77, %cneg1 : i32
    %tmp79 = arith.addi %tmp76, %tmp78 : i32
    %tmp80 = arith.addi %tmp79, %c0 : i32
    %tmp81 = arith.muli %tmp80, %tmp77 : i32
    %tmp82 = arith.xori %tmp81, %cneg1 : i32
    %tmp83 = arith.addi %tmp80, %tmp82 : i32
    %tmp84 = arith.addi %tmp83, %c0 : i32
    %tmp85 = arith.addi %tmp84, %c0 : i32
    
    // Final merge with D
    %tmp87 = arith.addi %tmp85, %d1 : i32
    
    %c0_idx = arith.constant 0 : index
    memref.store %tmp87, %arg0[%c0_idx] : memref<?xi32>
    cf.br ^bb1(%tmp87, %d1 : i32, i32)
}

// CHECK:           func.func @infinte_loop_v2(%[[ARG0:.*]]: memref<?xi32>, %[[ARG1:.*]]: i32) {
// CHECK:           %vl15969$e5677$ = arith.constant 1 : i32
// CHECK:           %vl15390$funcArg1-vl15969$ = arith.addi %[[ARG1]], %vl15969$e5677$ : i32
// CHECK:           cf.br ^bb1(%vl15390$funcArg1-vl15969$, %vl15390$funcArg1-vl15969$ : i32, i32)
// CHECK:           ^bb1(%0: i32, %1: i32):
// CHECK:           %vl85743$20b04$ = arith.constant 0 : i32
// CHECK:           %vl73800$blockArg0-vl85743$ = arith.muli %0, %vl85743$20b04$ : i32
// CHECK:           %vl85743$ded78$ = arith.constant -1 : i32
// CHECK:           %op51214$vl73800-vl85743$ = arith.xori %vl73800$blockArg0-vl85743$, %vl85743$ded78$ : i32
// CHECK:           %op12693$blockArg0-op51214$ = arith.addi %0, %op51214$vl73800-vl85743$ : i32
// CHECK:           %vl34407$blockArg1-vl85743$ = arith.addi %1, %vl85743$ded78$ : i32
// CHECK:           %op15672$op12693-vl34407$ = arith.addi %op12693$blockArg0-op51214$, %vl34407$blockArg1-vl85743$ : i32
// CHECK:           %op97825$op15672-vl73800$ = arith.muli %op15672$op12693-vl34407$, %vl73800$blockArg0-vl85743$ : i32
// CHECK:           %op51214$op97825-vl85743$ = arith.xori %op97825$op15672-vl73800$, %vl85743$ded78$ : i32
// CHECK:           %op12343$op15672-op51214$ = arith.addi %op15672$op12693-vl34407$, %op51214$op97825-vl85743$ : i32
// CHECK:           %op27844$op12343-vl85743$ = arith.addi %op12343$op15672-op51214$, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$ = arith.muli %op27844$op12343-vl85743$, %op97825$op15672-vl73800$ : i32
// CHECK:           %op51214$op97825-vl85743$_0 = arith.xori %op97825$op27844-op97825$, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$ = arith.addi %op27844$op12343-vl85743$, %op51214$op97825-vl85743$_0 : i32
// CHECK:           %op27844$op12343-vl85743$_1 = arith.addi %op12343$op27844-op51214$, %vl85743$20b04$ : i32
// CHECK:           %op27844$op27844-vl85743$ = arith.addi %op27844$op12343-vl85743$_1, %vl85743$20b04$ : i32
// CHECK:           %op27844$op27844-vl85743$_2 = arith.addi %op27844$op27844-vl85743$, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_3 = arith.muli %op27844$op12343-vl85743$_1, %op97825$op27844-op97825$ : i32
// CHECK:           %op97825$op27844-op97825$_4 = arith.muli %op27844$op27844-vl85743$_2, %op97825$op27844-op97825$_3 : i32
// CHECK:           %op51214$op97825-vl85743$_5 = arith.xori %op97825$op27844-op97825$_4, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_6 = arith.addi %op27844$op27844-vl85743$_2, %op51214$op97825-vl85743$_5 : i32
// CHECK:           %op27844$op12343-vl85743$_7 = arith.addi %op12343$op27844-op51214$_6, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_8 = arith.muli %op27844$op12343-vl85743$_7, %op97825$op27844-op97825$_4 : i32
// CHECK:           %op51214$op97825-vl85743$_9 = arith.xori %op97825$op27844-op97825$_8, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_10 = arith.addi %op27844$op12343-vl85743$_7, %op51214$op97825-vl85743$_9 : i32
// CHECK:           %op27844$op12343-vl85743$_11 = arith.addi %op12343$op27844-op51214$_10, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_12 = arith.muli %op27844$op12343-vl85743$_11, %op97825$op27844-op97825$_8 : i32
// CHECK:           %op51214$op97825-vl85743$_13 = arith.xori %op97825$op27844-op97825$_12, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_14 = arith.addi %op27844$op12343-vl85743$_11, %op51214$op97825-vl85743$_13 : i32
// CHECK:           %op27844$op12343-vl85743$_15 = arith.addi %op12343$op27844-op51214$_14, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_16 = arith.muli %op27844$op12343-vl85743$_15, %op97825$op27844-op97825$_12 : i32
// CHECK:           %op51214$op97825-vl85743$_17 = arith.xori %op97825$op27844-op97825$_16, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_18 = arith.addi %op27844$op12343-vl85743$_15, %op51214$op97825-vl85743$_17 : i32
// CHECK:           %op27844$op12343-vl85743$_19 = arith.addi %op12343$op27844-op51214$_18, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_20 = arith.muli %op27844$op12343-vl85743$_19, %op97825$op27844-op97825$_16 : i32
// CHECK:           %op51214$op97825-vl85743$_21 = arith.xori %op97825$op27844-op97825$_20, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_22 = arith.addi %op27844$op12343-vl85743$_19, %op51214$op97825-vl85743$_21 : i32
// CHECK:           %vl85743$51850$ = arith.constant -9 : i32
// CHECK:           %vl34407$blockArg1-vl85743$_23 = arith.addi %1, %vl85743$51850$ : i32
// CHECK:           %op17008$vl34407-vl85743$ = arith.muli %vl34407$blockArg1-vl85743$_23, %vl85743$20b04$ : i32
// CHECK:           %op51214$op17008-vl85743$ = arith.xori %op17008$vl34407-vl85743$, %vl85743$ded78$ : i32
// CHECK:           %op12343$op51214-vl34407$ = arith.addi %op51214$op17008-vl85743$, %vl34407$blockArg1-vl85743$_23 : i32
// CHECK:           %op15672$op12343-op12343$ = arith.addi %op12343$op27844-op51214$_22, %op12343$op51214-vl34407$ : i32
// CHECK:           %op97825$op15672-op97825$ = arith.muli %op15672$op12343-op12343$, %op97825$op27844-op97825$_20 : i32
// CHECK:           %op51214$op97825-vl85743$_24 = arith.xori %op97825$op15672-op97825$, %vl85743$ded78$ : i32
// CHECK:           %op12343$op15672-op51214$_25 = arith.addi %op15672$op12343-op12343$, %op51214$op97825-vl85743$_24 : i32
// CHECK:           %op17008$vl34407-vl85743$_26 = arith.muli %vl34407$blockArg1-vl85743$, %vl85743$20b04$ : i32
// CHECK:           %op51214$op17008-vl85743$_27 = arith.xori %op17008$vl34407-vl85743$_26, %vl85743$ded78$ : i32
// CHECK:           %op12343$op51214-vl34407$_28 = arith.addi %op51214$op17008-vl85743$_27, %vl34407$blockArg1-vl85743$ : i32
// CHECK:           %op97825$op12343-op17008$ = arith.muli %op12343$op51214-vl34407$_28, %op17008$vl34407-vl85743$_26 : i32
// CHECK:           %op51214$op97825-vl85743$_29 = arith.xori %op97825$op12343-op17008$, %vl85743$ded78$ : i32
// CHECK:           %op12343$op12343-op51214$ = arith.addi %op12343$op51214-vl34407$_28, %op51214$op97825-vl85743$_29 : i32
// CHECK:           %op27844$op12343-vl85743$_30 = arith.addi %op12343$op12343-op51214$, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_31 = arith.muli %op27844$op12343-vl85743$_30, %op97825$op12343-op17008$ : i32
// CHECK:           %op51214$op97825-vl85743$_32 = arith.xori %op97825$op27844-op97825$_31, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_33 = arith.addi %op27844$op12343-vl85743$_30, %op51214$op97825-vl85743$_32 : i32
// CHECK:           %op15672$op12343-op12343$_34 = arith.addi %op12343$op15672-op51214$_25, %op12343$op27844-op51214$_33 : i32
// CHECK:           %op97825$op15672-op97825$_35 = arith.muli %op15672$op12343-op12343$_34, %op97825$op15672-op97825$ : i32
// CHECK:           %op51214$op97825-vl85743$_36 = arith.xori %op97825$op15672-op97825$_35, %vl85743$ded78$ : i32
// CHECK:           %op12343$op15672-op51214$_37 = arith.addi %op15672$op12343-op12343$_34, %op51214$op97825-vl85743$_36 : i32
// CHECK:           %op27844$op12343-vl85743$_38 = arith.addi %op12343$op15672-op51214$_37, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_39 = arith.muli %op27844$op12343-vl85743$_38, %op97825$op15672-op97825$_35 : i32
// CHECK:           %op51214$op97825-vl85743$_40 = arith.xori %op97825$op27844-op97825$_39, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_41 = arith.addi %op27844$op12343-vl85743$_38, %op51214$op97825-vl85743$_40 : i32
// CHECK:           %op27844$op12343-vl85743$_42 = arith.addi %op12343$op27844-op51214$_41, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_43 = arith.muli %op27844$op12343-vl85743$_42, %op97825$op27844-op97825$_39 : i32
// CHECK:           %op51214$op97825-vl85743$_44 = arith.xori %op97825$op27844-op97825$_43, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_45 = arith.addi %op27844$op12343-vl85743$_42, %op51214$op97825-vl85743$_44 : i32
// CHECK:           %vl85743$7b7de$ = arith.constant -14 : i32
// CHECK:           %vl34407$blockArg1-vl85743$_46 = arith.addi %1, %vl85743$7b7de$ : i32
// CHECK:           %op17008$vl34407-vl85743$_47 = arith.muli %vl34407$blockArg1-vl85743$_46, %vl85743$20b04$ : i32
// CHECK:           %op51214$op17008-vl85743$_48 = arith.xori %op17008$vl34407-vl85743$_47, %vl85743$ded78$ : i32
// CHECK:           %op12343$op51214-vl34407$_49 = arith.addi %op51214$op17008-vl85743$_48, %vl34407$blockArg1-vl85743$_46 : i32
// CHECK:           %op97825$op12343-op17008$_50 = arith.muli %op12343$op51214-vl34407$_49, %op17008$vl34407-vl85743$_47 : i32
// CHECK:           %op51214$op97825-vl85743$_51 = arith.xori %op97825$op12343-op17008$_50, %vl85743$ded78$ : i32
// CHECK:           %op12343$op12343-op51214$_52 = arith.addi %op12343$op51214-vl34407$_49, %op51214$op97825-vl85743$_51 : i32
// CHECK:           %op27844$op12343-vl85743$_53 = arith.addi %op12343$op12343-op51214$_52, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_54 = arith.muli %op27844$op12343-vl85743$_53, %op97825$op12343-op17008$_50 : i32
// CHECK:           %op51214$op97825-vl85743$_55 = arith.xori %op97825$op27844-op97825$_54, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_56 = arith.addi %op27844$op12343-vl85743$_53, %op51214$op97825-vl85743$_55 : i32
// CHECK:           %op27844$op12343-vl85743$_57 = arith.addi %op12343$op27844-op51214$_56, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_58 = arith.muli %op27844$op12343-vl85743$_57, %op97825$op27844-op97825$_54 : i32
// CHECK:           %op51214$op97825-vl85743$_59 = arith.xori %op97825$op27844-op97825$_58, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_60 = arith.addi %op27844$op12343-vl85743$_57, %op51214$op97825-vl85743$_59 : i32
// CHECK:           %op27844$op12343-vl85743$_61 = arith.addi %op12343$op27844-op51214$_60, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_62 = arith.muli %op27844$op12343-vl85743$_61, %op97825$op27844-op97825$_58 : i32
// CHECK:           %op51214$op97825-vl85743$_63 = arith.xori %op97825$op27844-op97825$_62, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_64 = arith.addi %op27844$op12343-vl85743$_61, %op51214$op97825-vl85743$_63 : i32
// CHECK:           %op27844$op12343-vl85743$_65 = arith.addi %op12343$op27844-op51214$_64, %vl85743$20b04$ : i32
// CHECK:           %op15672$op12343-op27844$ = arith.addi %op12343$op27844-op51214$_45, %op27844$op12343-vl85743$_65 : i32
// CHECK:           %op97825$op15672-op97825$_66 = arith.muli %op15672$op12343-op27844$, %op97825$op27844-op97825$_43 : i32
// CHECK:           %op51214$op97825-vl85743$_67 = arith.xori %op97825$op15672-op97825$_66, %vl85743$ded78$ : i32
// CHECK:           %op12343$op15672-op51214$_68 = arith.addi %op15672$op12343-op27844$, %op51214$op97825-vl85743$_67 : i32
// CHECK:           %op15672$op12343-vl34407$ = arith.addi %op12343$op15672-op51214$_68, %vl34407$blockArg1-vl85743$_46 : i32
// CHECK:           %op97825$op15672-op97825$_69 = arith.muli %op15672$op12343-vl34407$, %op97825$op15672-op97825$_66 : i32
// CHECK:           %op51214$op97825-vl85743$_70 = arith.xori %op97825$op15672-op97825$_69, %vl85743$ded78$ : i32
// CHECK:           %op12343$op15672-op51214$_71 = arith.addi %op15672$op12343-vl34407$, %op51214$op97825-vl85743$_70 : i32
// CHECK:           %vl85743$1e72e$ = arith.constant -21 : i32
// CHECK:           %vl34407$blockArg1-vl85743$_72 = arith.addi %1, %vl85743$1e72e$ : i32
// CHECK:           %op17008$vl34407-vl85743$_73 = arith.muli %vl34407$blockArg1-vl85743$_72, %vl85743$20b04$ : i32
// CHECK:           %op51214$op17008-vl85743$_74 = arith.xori %op17008$vl34407-vl85743$_73, %vl85743$ded78$ : i32
// CHECK:           %op12343$op51214-vl34407$_75 = arith.addi %op51214$op17008-vl85743$_74, %vl34407$blockArg1-vl85743$_72 : i32
// CHECK:           %op97825$op12343-op17008$_76 = arith.muli %op12343$op51214-vl34407$_75, %op17008$vl34407-vl85743$_73 : i32
// CHECK:           %op51214$op97825-vl85743$_77 = arith.xori %op97825$op12343-op17008$_76, %vl85743$ded78$ : i32
// CHECK:           %op12343$op12343-op51214$_78 = arith.addi %op12343$op51214-vl34407$_75, %op51214$op97825-vl85743$_77 : i32
// CHECK:           %op27844$op12343-vl85743$_79 = arith.addi %op12343$op12343-op51214$_78, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_80 = arith.muli %op27844$op12343-vl85743$_79, %op97825$op12343-op17008$_76 : i32
// CHECK:           %op51214$op97825-vl85743$_81 = arith.xori %op97825$op27844-op97825$_80, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_82 = arith.addi %op27844$op12343-vl85743$_79, %op51214$op97825-vl85743$_81 : i32
// CHECK:           %op27844$op12343-vl85743$_83 = arith.addi %op12343$op27844-op51214$_82, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_84 = arith.muli %op27844$op12343-vl85743$_83, %op97825$op27844-op97825$_80 : i32
// CHECK:           %op51214$op97825-vl85743$_85 = arith.xori %op97825$op27844-op97825$_84, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_86 = arith.addi %op27844$op12343-vl85743$_83, %op51214$op97825-vl85743$_85 : i32
// CHECK:           %op27844$op12343-vl85743$_87 = arith.addi %op12343$op27844-op51214$_86, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_88 = arith.muli %op27844$op12343-vl85743$_87, %op97825$op27844-op97825$_84 : i32
// CHECK:           %op51214$op97825-vl85743$_89 = arith.xori %op97825$op27844-op97825$_88, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_90 = arith.addi %op27844$op12343-vl85743$_87, %op51214$op97825-vl85743$_89 : i32
// CHECK:           %op27844$op12343-vl85743$_91 = arith.addi %op12343$op27844-op51214$_90, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_92 = arith.muli %op27844$op12343-vl85743$_91, %op97825$op27844-op97825$_88 : i32
// CHECK:           %op13782$op12343-op97825$ = arith.addi %op12343$op15672-op51214$_71, %op97825$op27844-op97825$_92 : i32
// CHECK:           %op97825$op13782-op97825$ = arith.muli %op13782$op12343-op97825$, %op97825$op15672-op97825$_69 : i32
// CHECK:           %op51214$op97825-vl85743$_93 = arith.xori %op97825$op13782-op97825$, %vl85743$ded78$ : i32
// CHECK:           %op12343$op13782-op51214$ = arith.addi %op13782$op12343-op97825$, %op51214$op97825-vl85743$_93 : i32
// CHECK:           %op27844$op12343-vl85743$_94 = arith.addi %op12343$op13782-op51214$, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_95 = arith.muli %op27844$op12343-vl85743$_94, %op97825$op13782-op97825$ : i32
// CHECK:           %op51214$op97825-vl85743$_96 = arith.xori %op97825$op27844-op97825$_95, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_97 = arith.addi %op27844$op12343-vl85743$_94, %op51214$op97825-vl85743$_96 : i32
// CHECK:           %op27844$op12343-vl85743$_98 = arith.addi %op12343$op27844-op51214$_97, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_99 = arith.muli %op27844$op12343-vl85743$_98, %op97825$op27844-op97825$_95 : i32
// CHECK:           %op51214$op97825-vl85743$_100 = arith.xori %op97825$op27844-op97825$_99, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_101 = arith.addi %op27844$op12343-vl85743$_98, %op51214$op97825-vl85743$_100 : i32
// CHECK:           %op27844$op12343-vl85743$_102 = arith.addi %op12343$op27844-op51214$_101, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_103 = arith.muli %op27844$op12343-vl85743$_102, %op97825$op27844-op97825$_99 : i32
// CHECK:           %op51214$op97825-vl85743$_104 = arith.xori %op97825$op27844-op97825$_103, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_105 = arith.addi %op27844$op12343-vl85743$_102, %op51214$op97825-vl85743$_104 : i32
// CHECK:           %op27844$op12343-vl85743$_106 = arith.addi %op12343$op27844-op51214$_105, %vl85743$20b04$ : i32
// CHECK:           %op97825$op27844-op97825$_107 = arith.muli %op27844$op12343-vl85743$_106, %op97825$op27844-op97825$_103 : i32
// CHECK:           %op51214$op97825-vl85743$_108 = arith.xori %op97825$op27844-op97825$_107, %vl85743$ded78$ : i32
// CHECK:           %op12343$op27844-op51214$_109 = arith.addi %op27844$op12343-vl85743$_106, %op51214$op97825-vl85743$_108 : i32
// CHECK:           %op27844$op12343-vl85743$_110 = arith.addi %op12343$op27844-op51214$_109, %vl85743$20b04$ : i32
// CHECK:           %op27844$op27844-vl85743$_111 = arith.addi %op27844$op12343-vl85743$_110, %vl85743$20b04$ : i32
// CHECK:           %op15672$op27844-vl34407$ = arith.addi %op27844$op27844-vl85743$_111, %vl34407$blockArg1-vl85743$_72 : i32
// CHECK:           %vl16744$e527e$ = arith.constant 0 : index
// CHECK:           memref.store %op15672$op27844-vl34407$, %[[ARG0]][%vl16744$e527e$] : memref<?xi32>
// CHECK:           cf.br ^bb1(%op15672$op27844-vl34407$, %vl34407$blockArg1-vl85743$_72 : i32, i32)
// CHECK:         }
func.func @infinte_loop_v2(%arg0: memref<?xi32>, %arg1: i32) {
    %c1 = arith.constant 1 : i32
    %a = arith.addi %arg1, %c1 : i32
    cf.br ^bb1(%a, %a : i32, i32)

    ^bb1(%tmp: i32, %tmp2: i32):
    %c0 = arith.constant 0 : i32
    %cneg1 = arith.constant -1 : i32
    %cneg9 = arith.constant -9 : i32
    %cneg14 = arith.constant -14 : i32
    %cneg21 = arith.constant -21 : i32
    
    // Branch D: Start with cneg21
    %d1 = arith.addi %tmp2, %cneg21 : i32
    
    // Branch C: tmp2-cneg9 chain
    %base1 = arith.addi %tmp2, %cneg9 : i32
    %base2 = arith.muli %base1, %c0 : i32
    
    // Branch A: First main chain
    %tmp3 = arith.muli %tmp, %c0 : i32
    
    // Branch D continued
    %d2 = arith.muli %d1, %c0 : i32
    %d3 = arith.xori %d2, %cneg1 : i32
    %d4 = arith.addi %d1, %d3 : i32
    
    // Branch E: cneg14 chain
    %e1 = arith.addi %tmp2, %cneg14 : i32
    %e2 = arith.muli %e1, %c0 : i32
    
    // Branch A continued
    %tmp4 = arith.xori %tmp3, %cneg1 : i32
    %tmp5 = arith.addi %tmp, %tmp4 : i32
    
    // Branch B: tmp6 chain
    %tmp6 = arith.addi %tmp2, %cneg1 : i32
    %alt1 = arith.muli %tmp6, %c0 : i32
    
    // Branch C continued
    %base3 = arith.xori %base2, %cneg1 : i32
    
    // Branch D continued
    %d5 = arith.muli %d4, %d2 : i32
    %d6 = arith.xori %d5, %cneg1 : i32
    %d7 = arith.addi %d4, %d6 : i32
    
    // Branch E continued
    %e3 = arith.xori %e2, %cneg1 : i32
    %e4 = arith.addi %e1, %e3 : i32
    %e5 = arith.muli %e4, %e2 : i32
    
    // Branch A continued
    %tmp7 = arith.addi %tmp5, %tmp6 : i32
    %tmp8 = arith.muli %tmp7, %tmp3 : i32
    
    // Branch B continued
    %alt2 = arith.xori %alt1, %cneg1 : i32
    %alt3 = arith.addi %tmp6, %alt2 : i32
    %alt4 = arith.muli %alt3, %alt1 : i32
    
    // Branch C continued
    %base4 = arith.addi %base1, %base3 : i32
    
    // Branch D continued
    %d8 = arith.addi %d7, %c0 : i32
    %d9 = arith.muli %d8, %d5 : i32
    
    // Branch A continued
    %tmp9 = arith.xori %tmp8, %cneg1 : i32
    %tmp10 = arith.addi %tmp7, %tmp9 : i32
    %tmp11 = arith.addi %tmp10, %c0 : i32
    
    // Branch E continued
    %e6 = arith.xori %e5, %cneg1 : i32
    %e7 = arith.addi %e4, %e6 : i32
    %e8 = arith.addi %e7, %c0 : i32
    
    // Branch B continued
    %alt5 = arith.xori %alt4, %cneg1 : i32
    %alt6 = arith.addi %alt3, %alt5 : i32
    
    // Branch A continued
    %tmp12 = arith.muli %tmp11, %tmp8 : i32
    %tmp13 = arith.xori %tmp12, %cneg1 : i32
    
    // Branch D continued
    %d10 = arith.xori %d9, %cneg1 : i32
    %d11 = arith.addi %d8, %d10 : i32
    %d12 = arith.addi %d11, %c0 : i32
    
    // Branch E continued
    %e9 = arith.muli %e8, %e5 : i32
    %e10 = arith.xori %e9, %cneg1 : i32
    
    // Branch B continued
    %alt7 = arith.addi %alt6, %c0 : i32
    %alt8 = arith.muli %alt7, %alt4 : i32
    %alt9 = arith.xori %alt8, %cneg1 : i32
    
    // Branch A continued
    %tmp14 = arith.addi %tmp11, %tmp13 : i32
    %tmp15 = arith.addi %tmp14, %c0 : i32
    %tmp16 = arith.muli %tmp15, %tmp12 : i32
    
    // Branch D continued
    %d13 = arith.muli %d12, %d9 : i32
    %d14 = arith.xori %d13, %cneg1 : i32
    %d15 = arith.addi %d12, %d14 : i32
    
    // Branch E continued
    %e11 = arith.addi %e8, %e10 : i32
    %e12 = arith.addi %e11, %c0 : i32
    %e13 = arith.muli %e12, %e9 : i32
    
    // Branch A continued
    %tmp17 = arith.addi %tmp15, %c0 : i32
    %tmp18 = arith.addi %tmp17, %c0 : i32
    %tmp19 = arith.muli %tmp18, %tmp16 : i32
    
    // Branch B continued
    %alt10 = arith.addi %alt7, %alt9 : i32
    
    // Branch D continued
    %d16 = arith.addi %d15, %c0 : i32
    %d17 = arith.muli %d16, %d13 : i32
    %d18 = arith.xori %d17, %cneg1 : i32
    
    // Branch A continued
    %tmp20 = arith.xori %tmp19, %cneg1 : i32
    %tmp21 = arith.addi %tmp18, %tmp20 : i32
    %tmp22 = arith.addi %tmp21, %c0 : i32
    
    // Branch E continued
    %e14 = arith.xori %e13, %cneg1 : i32
    %e15 = arith.addi %e12, %e14 : i32
    %e16 = arith.addi %e15, %c0 : i32
    
    // Branch A continued
    %tmp23 = arith.muli %tmp22, %tmp19 : i32
    %tmp24 = arith.xori %tmp23, %cneg1 : i32
    
    // Branch D continued
    %d19 = arith.addi %d16, %d18 : i32
    %d20 = arith.addi %d19, %c0 : i32
    %d21 = arith.muli %d20, %d17 : i32
    
    // Branch A continued
    %tmp25 = arith.addi %tmp22, %tmp24 : i32
    %tmp26 = arith.addi %tmp25, %c0 : i32
    
    // Branch E continued
    %e17 = arith.muli %e16, %e13 : i32
    %e18 = arith.xori %e17, %cneg1 : i32
    %e19 = arith.addi %e16, %e18 : i32
    %e20 = arith.addi %e19, %c0 : i32
    
    // Branch A continued
    %tmp27 = arith.muli %tmp26, %tmp23 : i32
    %tmp28 = arith.xori %tmp27, %cneg1 : i32
    %tmp29 = arith.addi %tmp26, %tmp28 : i32
    %tmp30 = arith.addi %tmp29, %c0 : i32
    %tmp31 = arith.muli %tmp30, %tmp27 : i32
    %tmp32 = arith.xori %tmp31, %cneg1 : i32
    %tmp33 = arith.addi %tmp30, %tmp32 : i32
    %tmp34 = arith.addi %tmp33, %c0 : i32
    %tmp35 = arith.muli %tmp34, %tmp31 : i32
    %tmp36 = arith.xori %tmp35, %cneg1 : i32
    %tmp37 = arith.addi %tmp34, %tmp36 : i32
    
    // Merge A + C
    %tmp39 = arith.addi %tmp37, %base4 : i32
    %tmp40 = arith.muli %tmp39, %tmp35 : i32
    %tmp41 = arith.xori %tmp40, %cneg1 : i32
    %tmp42 = arith.addi %tmp39, %tmp41 : i32
    
    // Merge + B
    %tmp43 = arith.addi %tmp42, %alt10 : i32
    %tmp44 = arith.muli %tmp43, %tmp40 : i32
    %tmp45 = arith.xori %tmp44, %cneg1 : i32
    %tmp46 = arith.addi %tmp43, %tmp45 : i32
    %tmp47 = arith.addi %tmp46, %c0 : i32
    %tmp48 = arith.muli %tmp47, %tmp44 : i32
    %tmp49 = arith.xori %tmp48, %cneg1 : i32
    %tmp50 = arith.addi %tmp47, %tmp49 : i32
    %tmp51 = arith.addi %tmp50, %c0 : i32
    
    // Merge + E
    %tmp52 = arith.muli %tmp51, %tmp48 : i32
    %tmp53 = arith.xori %tmp52, %cneg1 : i32
    %tmp54 = arith.addi %tmp51, %tmp53 : i32
    %tmp55 = arith.addi %tmp54, %e20 : i32
    %tmp56 = arith.muli %tmp55, %tmp52 : i32
    %tmp57 = arith.xori %tmp56, %cneg1 : i32
    %tmp58 = arith.addi %tmp55, %tmp57 : i32
    
    // Merge E source value
    %tmp60 = arith.addi %tmp58, %e1 : i32
    %tmp61 = arith.muli %tmp60, %tmp56 : i32
    %tmp62 = arith.xori %tmp61, %cneg1 : i32
    %tmp63 = arith.addi %tmp60, %tmp62 : i32
    
    // Merge + D
    %tmp64 = arith.addi %tmp63, %d21 : i32
    %tmp65 = arith.muli %tmp64, %tmp61 : i32
    %tmp66 = arith.xori %tmp65, %cneg1 : i32
    %tmp67 = arith.addi %tmp64, %tmp66 : i32
    %tmp68 = arith.addi %tmp67, %c0 : i32
    %tmp69 = arith.muli %tmp68, %tmp65 : i32
    %tmp70 = arith.xori %tmp69, %cneg1 : i32
    %tmp71 = arith.addi %tmp68, %tmp70 : i32
    %tmp72 = arith.addi %tmp71, %c0 : i32
    %tmp73 = arith.muli %tmp72, %tmp69 : i32
    %tmp74 = arith.xori %tmp73, %cneg1 : i32
    %tmp75 = arith.addi %tmp72, %tmp74 : i32
    %tmp76 = arith.addi %tmp75, %c0 : i32
    %tmp77 = arith.muli %tmp76, %tmp73 : i32
    %tmp78 = arith.xori %tmp77, %cneg1 : i32
    %tmp79 = arith.addi %tmp76, %tmp78 : i32
    %tmp80 = arith.addi %tmp79, %c0 : i32
    %tmp81 = arith.muli %tmp80, %tmp77 : i32
    %tmp82 = arith.xori %tmp81, %cneg1 : i32
    %tmp83 = arith.addi %tmp80, %tmp82 : i32
    %tmp84 = arith.addi %tmp83, %c0 : i32
    %tmp85 = arith.addi %tmp84, %c0 : i32
    
    // Final merge with D source value
    %tmp87 = arith.addi %tmp85, %d1 : i32
    
    %c0_idx = arith.constant 0 : index
    memref.store %tmp87, %arg0[%c0_idx] : memref<?xi32>
    cf.br ^bb1(%tmp87, %d1 : i32, i32)
}

// CHECK:       }
