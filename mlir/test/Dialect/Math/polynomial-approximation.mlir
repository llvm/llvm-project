// RUN: mlir-opt %s -test-math-polynomial-approximation | FileCheck %s
// RUN: mlir-opt %s -test-math-polynomial-approximation=enable-avx2 \
// RUN: | FileCheck --check-prefix=AVX2 %s

// Check that all math functions lowered to approximations built from
// standard operations (add, mul, fma, shift, etc...).

// CHECK-LABEL: func @erf_scalar(
// CHECK-SAME:    %[[val_arg0:.*]]: f32) -> f32 {
// CHECK-DAG:     %[[val_cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[val_cst_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:     %[[val_cst_1:.*]] = arith.constant 1.12837911 : f32
// CHECK-DAG:     %[[val_cst_2:.*]] = arith.constant -0.523018539 : f32
// CHECK-DAG:     %[[val_cst_3:.*]] = arith.constant 0.209741712 : f32
// CHECK-DAG:     %[[val_cst_4:.*]] = arith.constant 0.0258146804 : f32
// CHECK-DAG:     %[[val_cst_5:.*]] = arith.constant 1.12750685 : f32
// CHECK-DAG:     %[[val_cst_6:.*]] = arith.constant -0.364721417 : f32
// CHECK-DAG:     %[[val_cst_7:.*]] = arith.constant 0.118407398 : f32
// CHECK-DAG:     %[[val_cst_8:.*]] = arith.constant 0.0370645523 : f32
// CHECK-DAG:     %[[val_cst_9:.*]] = arith.constant -0.00330093061 : f32
// CHECK-DAG:     %[[val_cst_10:.*]] = arith.constant 0.00351961935 : f32
// CHECK-DAG:     %[[val_cst_11:.*]] = arith.constant -0.00141373626 : f32
// CHECK-DAG:     %[[val_cst_12:.*]] = arith.constant 2.53447099E-4 : f32
// CHECK-DAG:     %[[val_cst_13:.*]] = arith.constant -1.71048032E-5 : f32
// CHECK-DAG:     %[[val_cst_14:.*]] = arith.constant -0.463513821 : f32
// CHECK-DAG:     %[[val_cst_15:.*]] = arith.constant 0.519230127 : f32
// CHECK-DAG:     %[[val_cst_16:.*]] = arith.constant -0.131808966 : f32
// CHECK-DAG:     %[[val_cst_17:.*]] = arith.constant 0.0739796459 : f32
// CHECK-DAG:     %[[val_cst_18:.*]] = arith.constant -3.276070e-01 : f32
// CHECK-DAG:     %[[val_cst_19:.*]] = arith.constant 0.448369086 : f32
// CHECK-DAG:     %[[val_cst_20:.*]] = arith.constant -0.0883462652 : f32
// CHECK-DAG:     %[[val_cst_21:.*]] = arith.constant 0.0572442785 : f32
// CHECK-DAG:     %[[val_cst_22:.*]] = arith.constant -2.0606916 : f32
// CHECK-DAG:     %[[val_cst_23:.*]] = arith.constant 1.62705934 : f32
// CHECK-DAG:     %[[val_cst_24:.*]] = arith.constant -0.583389878 : f32
// CHECK-DAG:     %[[val_cst_25:.*]] = arith.constant 0.0821908935 : f32
// CHECK-DAG:     %[[val_cst_26:.*]] = arith.constant 8.000000e-01 : f32
// CHECK-DAG:     %[[val_cst_27:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:     %[[val_cst_28:.*]] = arith.constant 3.750000e+00 : f32
// CHECK:         %[[val_0:.*]] = arith.cmpf olt, %[[val_arg0]], %[[val_cst]] : f32
// CHECK:         %[[val_1:.*]] = arith.negf %[[val_arg0]] : f32
// CHECK:         %[[val_2:.*]] = arith.select %[[val_0]], %[[val_1]], %[[val_arg0]] : f32
// CHECK:         %[[val_3:.*]] = arith.cmpf olt, %[[val_2]], %[[val_cst_26]] : f32
// CHECK:         %[[val_4:.*]] = arith.select %[[val_3]], %[[val_cst_1]], %[[val_cst_5]] : f32
// CHECK:         %[[val_5:.*]] = arith.select %[[val_3]], %[[val_cst_14]], %[[val_cst_18]] : f32
// CHECK:         %[[val_6:.*]] = arith.select %[[val_3]], %[[val_cst_2]], %[[val_cst_6]] : f32
// CHECK:         %[[val_7:.*]] = arith.select %[[val_3]], %[[val_cst_15]], %[[val_cst_19]] : f32
// CHECK:         %[[val_8:.*]] = arith.select %[[val_3]], %[[val_cst_3]], %[[val_cst_7]] : f32
// CHECK:         %[[val_9:.*]] = arith.select %[[val_3]], %[[val_cst_16]], %[[val_cst_20]] : f32
// CHECK:         %[[val_10:.*]] = arith.select %[[val_3]], %[[val_cst_4]], %[[val_cst_8]] : f32
// CHECK:         %[[val_11:.*]] = arith.select %[[val_3]], %[[val_cst_17]], %[[val_cst_21]] : f32
// CHECK:         %[[val_12:.*]] = arith.cmpf olt, %[[val_2]], %[[val_cst_27]] : f32
// CHECK:         %[[val_13:.*]] = arith.select %[[val_12]], %[[val_cst]], %[[val_cst_9]] : f32
// CHECK:         %[[val_14:.*]] = arith.select %[[val_12]], %[[val_4]], %[[val_cst_10]] : f32
// CHECK:         %[[val_15:.*]] = arith.select %[[val_12]], %[[val_5]], %[[val_cst_22]] : f32
// CHECK:         %[[val_16:.*]] = arith.select %[[val_12]], %[[val_6]], %[[val_cst_11]] : f32
// CHECK:         %[[val_17:.*]] = arith.select %[[val_12]], %[[val_7]], %[[val_cst_23]] : f32
// CHECK:         %[[val_18:.*]] = arith.select %[[val_12]], %[[val_8]], %[[val_cst_12]] : f32
// CHECK:         %[[val_19:.*]] = arith.select %[[val_12]], %[[val_9]], %[[val_cst_24]] : f32
// CHECK:         %[[val_20:.*]] = arith.select %[[val_12]], %[[val_10]], %[[val_cst_13]] : f32
// CHECK:         %[[val_21:.*]] = arith.select %[[val_12]], %[[val_11]], %[[val_cst_25]] : f32
// CHECK:         %[[val_22:.*]] = arith.select %[[val_12]], %[[val_cst]], %[[val_cst_0]] : f32
// CHECK:         %[[val_23:.*]] = arith.cmpf ult, %[[val_2]], %[[val_cst_28]] : f32
// CHECK:         %[[val_24:.*]] = math.fma %[[val_2]], %[[val_20]], %[[val_18]] : f32
// CHECK:         %[[val_25:.*]] = math.fma %[[val_2]], %[[val_24]], %[[val_16]] : f32
// CHECK:         %[[val_26:.*]] = math.fma %[[val_2]], %[[val_25]], %[[val_14]] : f32
// CHECK:         %[[val_27:.*]] = math.fma %[[val_2]], %[[val_26]], %[[val_13]] : f32
// CHECK:         %[[val_28:.*]] = math.fma %[[val_2]], %[[val_21]], %[[val_19]] : f32
// CHECK:         %[[val_29:.*]] = math.fma %[[val_2]], %[[val_28]], %[[val_17]] : f32
// CHECK:         %[[val_30:.*]] = math.fma %[[val_2]], %[[val_29]], %[[val_15]] : f32
// CHECK:         %[[val_31:.*]] = math.fma %[[val_2]], %[[val_30]], %[[val_cst_0]] : f32
// CHECK:         %[[val_32:.*]] = arith.divf %[[val_27]], %[[val_31]] : f32
// CHECK:         %[[val_33:.*]] = arith.addf %[[val_22]], %[[val_32]] : f32
// CHECK:         %[[val_34:.*]] = arith.select %[[val_23]], %[[val_33]], %[[val_cst_0]] : f32
// CHECK:         %[[val_35:.*]] = arith.negf %[[val_34]] : f32
// CHECK:         %[[val_36:.*]] = arith.select %[[val_0]], %[[val_35]], %[[val_34]] : f32
// CHECK:         return %[[val_36]] : f32
// CHECK:       }
func.func @erf_scalar(%arg0: f32) -> f32 {
  %0 = math.erf %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @erf_vector(
// CHECK-SAME:                     %[[arg0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:           %[[zero:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-NOT:       erf
// CHECK-COUNT-20:  select
// CHECK:           %[[res:.*]] = arith.select
// CHECK:           return %[[res]] : vector<8xf32>
// CHECK:         }
func.func @erf_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.erf %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL:   func @exp_scalar(
// CHECK-SAME:                     %[[VAL_0:.*]]: f32) -> f32 {
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 0.693147182 : f32
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1.44269502 : f32
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 0.499705136 : f32
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 0.168738902 : f32
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 0.0366896503 : f32
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 1.314350e-02 : f32
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 23 : i32
// CHECK-DAG:           %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:           %[[VAL_10:.*]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:           %[[VAL_11:.*]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:           %[[VAL_12:.*]] = arith.constant 1.17549435E-38 : f32
// CHECK-DAG:           %[[VAL_13:.*]] = arith.constant 127 : i32
// CHECK-DAG:           %[[VAL_14:.*]] = arith.constant -127 : i32
// CHECK:           %[[IS_NAN:.*]] = arith.cmpf uno, %[[VAL_0]], %[[VAL_0]] : f32
// CHECK:           %[[VAL_15:.*]] = arith.mulf %[[VAL_0]], %[[VAL_2]] : f32
// CHECK:           %[[VAL_16:.*]] = math.floor %[[VAL_15]] : f32
// CHECK:           %[[VAL_17:.*]] = arith.mulf %[[VAL_16]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_18:.*]] = arith.subf %[[VAL_0]], %[[VAL_17]] : f32
// CHECK:           %[[VAL_19:.*]] = arith.mulf %[[VAL_18]], %[[VAL_18]] : f32
// CHECK:           %[[VAL_20:.*]] = arith.mulf %[[VAL_19]], %[[VAL_19]] : f32
// CHECK:           %[[VAL_21:.*]] = math.fma %[[VAL_3]], %[[VAL_18]], %[[VAL_3]] : f32
// CHECK:           %[[VAL_22:.*]] = math.fma %[[VAL_5]], %[[VAL_18]], %[[VAL_4]] : f32
// CHECK:           %[[VAL_23:.*]] = math.fma %[[VAL_7]], %[[VAL_18]], %[[VAL_6]] : f32
// CHECK:           %[[VAL_24:.*]] = math.fma %[[VAL_22]], %[[VAL_19]], %[[VAL_21]] : f32
// CHECK:           %[[VAL_25:.*]] = math.fma %[[VAL_23]], %[[VAL_20]], %[[VAL_24]] : f32
// CHECK:           %[[VAL_26:.*]] = arith.fptosi %[[VAL_16]] : f32 to i32
// CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_13]] : i32
// CHECK:           %[[VAL_28:.*]] = arith.shli %[[VAL_27]], %[[VAL_8]] : i32
// CHECK:           %[[VAL_29:.*]] = arith.bitcast %[[VAL_28]] : i32 to f32
// CHECK:           %[[VAL_30:.*]] = arith.mulf %[[VAL_25]], %[[VAL_29]] : f32
// CHECK:           %[[VAL_31:.*]] = arith.cmpi sle, %[[VAL_26]], %[[VAL_13]] : i32
// CHECK:           %[[VAL_32:.*]] = arith.cmpi sge, %[[VAL_26]], %[[VAL_14]] : i32
// CHECK:           %[[VAL_33:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[VAL_11]] : f32
// CHECK:           %[[VAL_34:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[VAL_10]] : f32
// CHECK:           %[[VAL_35:.*]] = arith.cmpf ogt, %[[VAL_0]], %[[VAL_9]] : f32
// CHECK:           %[[VAL_36:.*]] = arith.andi %[[VAL_31]], %[[VAL_32]] : i1
// CHECK:           %[[VAL_37:.*]] = arith.select %[[VAL_35]], %[[VAL_10]], %[[VAL_12]] : f32
// CHECK:           %[[VAL_38:.*]] = arith.select %[[VAL_36]], %[[VAL_30]], %[[VAL_37]] : f32
// CHECK:           %[[VAL_39:.*]] = arith.select %[[VAL_34]], %[[VAL_10]], %[[VAL_38]] : f32
// CHECK:           %[[VAL_40:.*]] = arith.select %[[VAL_33]], %[[VAL_9]], %[[VAL_39]] : f32
// CHECK:           %[[VAL_41:.*]] = arith.select %[[IS_NAN]], %[[VAL_0]], %[[VAL_40]] : f32
// CHECK:           return %[[VAL_41]] : f32
func.func @exp_scalar(%arg0: f32) -> f32 {
  %0 = math.exp %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @exp_vector(
// CHECK-SAME:                     %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<0.693147182> : vector<8xf32>
// CHECK-NOT:       exp
// CHECK-COUNT-4:   select
// CHECK:           %[[VAL_40:.*]] = arith.select
// CHECK:           return %[[VAL_40]] : vector<8xf32>
func.func @exp_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.exp %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL:   func @expm1_scalar(
// CHECK-SAME:                       %[[X:.*]]: f32) -> f32 {
// CHECK-DAG:           %[[CST_MINUSONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG:           %[[CST_LOG2E:.*]] = arith.constant 1.44269502 : f32
// CHECK-DAG:           %[[CST_ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[BEGIN_EXP_X:.*]] = arith.mulf %[[X]], %[[CST_LOG2E]] : f32
// CHECK-NOT:       exp
// CHECK-COUNT-4:   select
// CHECK:           %[[EXP_X:.*]] = arith.select
// CHECK:           %[[IS_ONE_OR_NAN:.*]] = arith.cmpf ueq, %[[EXP_X]], %[[CST_ONE]] : f32
// CHECK:           %[[VAL_59:.*]] = arith.subf %[[EXP_X]], %[[CST_ONE]] : f32
// CHECK:           %[[VAL_60:.*]] = arith.cmpf oeq, %[[VAL_59]], %[[CST_MINUSONE]] : f32
// CHECK-NOT:       log
// CHECK-COUNT-5:   select
// CHECK:           %[[LOG_U:.*]] = arith.select
// CHECK:           %[[VAL_104:.*]] = arith.cmpf oeq, %[[LOG_U]], %[[EXP_X]] : f32
// CHECK:           %[[VAL_105:.*]] = arith.divf %[[X]], %[[LOG_U]] : f32
// CHECK:           %[[VAL_106:.*]] = arith.mulf %[[VAL_59]], %[[VAL_105]] : f32
// CHECK:           %[[VAL_107:.*]] = arith.select %[[VAL_104]], %[[EXP_X]], %[[VAL_106]] : f32
// CHECK:           %[[VAL_108:.*]] = arith.select %[[VAL_60]], %[[CST_MINUSONE]], %[[VAL_107]] : f32
// CHECK:           %[[VAL_109:.*]] = arith.select %[[IS_ONE_OR_NAN]], %[[X]], %[[VAL_108]] : f32
// CHECK:           return %[[VAL_109]] : f32
// CHECK:         }
func.func @expm1_scalar(%arg0: f32) -> f32 {
  %0 = math.expm1 %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @expm1_vector(
// CHECK-SAME:                       %[[VAL_0:.*]]: vector<8x8xf32>) -> vector<8x8xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<-1.000000e+00> : vector<8x8xf32>
// CHECK-NOT:       exp
// CHECK-COUNT-5:   select
// CHECK-NOT:       log
// CHECK-COUNT-5:   select
// CHECK-NOT:       expm1
// CHECK-COUNT-3:   select
// CHECK:           %[[VAL_115:.*]] = arith.select
// CHECK:           return %[[VAL_115]] : vector<8x8xf32>
// CHECK:         }
func.func @expm1_vector(%arg0: vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = math.expm1 %arg0 : vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// CHECK-LABEL:   func @log_scalar(
// CHECK-SAME:                             %[[X:.*]]: f32) -> f32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant -5.000000e-01 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 1.17549435E-38 : f32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0xFF800000 : f32
// CHECK:           %[[VAL_6:.*]] = arith.constant 0x7F800000 : f32
// CHECK:           %[[VAL_7:.*]] = arith.constant 0x7FC00000 : f32
// CHECK:           %[[VAL_8:.*]] = arith.constant 0.707106769 : f32
// CHECK:           %[[VAL_9:.*]] = arith.constant 0.0703768358 : f32
// CHECK:           %[[VAL_10:.*]] = arith.constant -0.115146101 : f32
// CHECK:           %[[VAL_11:.*]] = arith.constant 0.116769984 : f32
// CHECK:           %[[VAL_12:.*]] = arith.constant -0.12420141 : f32
// CHECK:           %[[VAL_13:.*]] = arith.constant 0.142493233 : f32
// CHECK:           %[[VAL_14:.*]] = arith.constant -0.166680574 : f32
// CHECK:           %[[VAL_15:.*]] = arith.constant 0.200007141 : f32
// CHECK:           %[[VAL_16:.*]] = arith.constant -0.24999994 : f32
// CHECK:           %[[VAL_17:.*]] = arith.constant 0.333333313 : f32
// CHECK:           %[[VAL_18:.*]] = arith.constant 1.260000e+02 : f32
// CHECK:           %[[VAL_19:.*]] = arith.constant -2139095041 : i32
// CHECK:           %[[VAL_20:.*]] = arith.constant 1056964608 : i32
// CHECK:           %[[VAL_21:.*]] = arith.constant 23 : i32
// CHECK:           %[[VAL_22:.*]] = arith.constant 0.693147182 : f32
// CHECK:           %[[VAL_23:.*]] = arith.cmpf ugt, %[[X]], %[[VAL_4]] : f32
// CHECK:           %[[VAL_24:.*]] = arith.select %[[VAL_23]], %[[X]], %[[VAL_4]] : f32
// CHECK-NOT:       frexp
// CHECK:           %[[VAL_25:.*]] = arith.bitcast %[[VAL_24]] : f32 to i32
// CHECK:           %[[VAL_26:.*]] = arith.andi %[[VAL_25]], %[[VAL_19]] : i32
// CHECK:           %[[VAL_27:.*]] = arith.ori %[[VAL_26]], %[[VAL_20]] : i32
// CHECK:           %[[VAL_28:.*]] = arith.bitcast %[[VAL_27]] : i32 to f32
// CHECK:           %[[VAL_29:.*]] = arith.bitcast %[[VAL_24]] : f32 to i32
// CHECK:           %[[VAL_30:.*]] = arith.shrui %[[VAL_29]], %[[VAL_21]] : i32
// CHECK:           %[[VAL_31:.*]] = arith.sitofp %[[VAL_30]] : i32 to f32
// CHECK:           %[[VAL_32:.*]] = arith.subf %[[VAL_31]], %[[VAL_18]] : f32
// CHECK:           %[[VAL_33:.*]] = arith.cmpf olt, %[[VAL_28]], %[[VAL_8]] : f32
// CHECK:           %[[VAL_34:.*]] = arith.select %[[VAL_33]], %[[VAL_28]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_35:.*]] = arith.subf %[[VAL_28]], %[[VAL_2]] : f32
// CHECK:           %[[VAL_36:.*]] = arith.select %[[VAL_33]], %[[VAL_2]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_37:.*]] = arith.subf %[[VAL_32]], %[[VAL_36]] : f32
// CHECK:           %[[VAL_38:.*]] = arith.addf %[[VAL_35]], %[[VAL_34]] : f32
// CHECK:           %[[VAL_39:.*]] = arith.mulf %[[VAL_38]], %[[VAL_38]] : f32
// CHECK:           %[[VAL_40:.*]] = arith.mulf %[[VAL_39]], %[[VAL_38]] : f32
// CHECK:           %[[VAL_41:.*]] = math.fma %[[VAL_9]], %[[VAL_38]], %[[VAL_10]] : f32
// CHECK:           %[[VAL_42:.*]] = math.fma %[[VAL_12]], %[[VAL_38]], %[[VAL_13]] : f32
// CHECK:           %[[VAL_43:.*]] = math.fma %[[VAL_15]], %[[VAL_38]], %[[VAL_16]] : f32
// CHECK:           %[[VAL_44:.*]] = math.fma %[[VAL_41]], %[[VAL_38]], %[[VAL_11]] : f32
// CHECK:           %[[VAL_45:.*]] = math.fma %[[VAL_42]], %[[VAL_38]], %[[VAL_14]] : f32
// CHECK:           %[[VAL_46:.*]] = math.fma %[[VAL_43]], %[[VAL_38]], %[[VAL_17]] : f32
// CHECK:           %[[VAL_47:.*]] = math.fma %[[VAL_44]], %[[VAL_40]], %[[VAL_45]] : f32
// CHECK:           %[[VAL_48:.*]] = math.fma %[[VAL_47]], %[[VAL_40]], %[[VAL_46]] : f32
// CHECK:           %[[VAL_49:.*]] = arith.mulf %[[VAL_48]], %[[VAL_40]] : f32
// CHECK:           %[[VAL_50:.*]] = math.fma %[[VAL_3]], %[[VAL_39]], %[[VAL_49]] : f32
// CHECK:           %[[VAL_51:.*]] = arith.addf %[[VAL_38]], %[[VAL_50]] : f32
// CHECK:           %[[VAL_52:.*]] = math.fma %[[VAL_37]], %[[VAL_22]], %[[VAL_51]] : f32
// CHECK:           %[[VAL_53:.*]] = arith.cmpf ult, %[[X]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_54:.*]] = arith.cmpf oeq, %[[X]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_55:.*]] = arith.cmpf oeq, %[[X]], %[[VAL_6]] : f32
// CHECK:           %[[VAL_56:.*]] = arith.select %[[VAL_55]], %[[VAL_6]], %[[VAL_52]] : f32
// CHECK:           %[[VAL_57:.*]] = arith.select %[[VAL_53]], %[[VAL_7]], %[[VAL_56]] : f32
// CHECK:           %[[VAL_58:.*]] = arith.select %[[VAL_54]], %[[VAL_5]], %[[VAL_57]] : f32
// CHECK:           return %[[VAL_58]] : f32
// CHECK:         }
func.func @log_scalar(%arg0: f32) -> f32 {
  %0 = math.log %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @log_vector(
// CHECK-SAME:                     %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:           %[[CST_LN2:.*]] = arith.constant dense<0.693147182> : vector<8xf32>
// CHECK-COUNT-5:   select
// CHECK:           %[[VAL_71:.*]] = arith.select
// CHECK:           return %[[VAL_71]] : vector<8xf32>
// CHECK:         }
func.func @log_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.log %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL:   func @log2_scalar(
// CHECK-SAME:                      %[[VAL_0:.*]]: f32) -> f32 {
// CHECK:           %[[CST_LOG2E:.*]] = arith.constant 1.44269502 : f32
// CHECK-COUNT-5:   select
// CHECK:           %[[VAL_65:.*]] = arith.select
// CHECK:           return %[[VAL_65]] : f32
// CHECK:         }
func.func @log2_scalar(%arg0: f32) -> f32 {
  %0 = math.log2 %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @log2_vector(
// CHECK-SAME:                      %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:           %[[CST_LOG2E:.*]] = arith.constant dense<1.44269502> : vector<8xf32>
// CHECK-COUNT-5:   select
// CHECK:           %[[VAL_71:.*]] = arith.select
// CHECK:           return %[[VAL_71]] : vector<8xf32>
// CHECK:         }
func.func @log2_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.log2 %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL:   func @log1p_scalar(
// CHECK-SAME:                       %[[X:.*]]: f32) -> f32 {
// CHECK:           %[[CST_ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[U:.*]] = arith.addf %[[X]], %[[CST_ONE]] : f32
// CHECK:           %[[U_SMALL:.*]] = arith.cmpf oeq, %[[U]], %[[CST_ONE]] : f32
// CHECK-NOT:       log
// CHECK-COUNT-5:   select
// CHECK:           %[[LOG_U:.*]] = arith.select
// CHECK:           %[[U_INF:.*]] = arith.cmpf oeq, %[[U]], %[[LOG_U]] : f32
// CHECK:           %[[VAL_69:.*]] = arith.subf %[[U]], %[[CST_ONE]] : f32
// CHECK:           %[[VAL_70:.*]] = arith.divf %[[LOG_U]], %[[VAL_69]] : f32
// CHECK:           %[[LOG_LARGE:.*]] = arith.mulf %[[X]], %[[VAL_70]] : f32
// CHECK:           %[[VAL_72:.*]] = arith.ori %[[U_SMALL]], %[[U_INF]]  : i1
// CHECK:           %[[APPROX:.*]] = arith.select %[[VAL_72]], %[[X]], %[[LOG_LARGE]] : f32
// CHECK:           return %[[APPROX]] : f32
// CHECK:         }
func.func @log1p_scalar(%arg0: f32) -> f32 {
  %0 = math.log1p %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @log1p_vector(
// CHECK-SAME:                       %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:           %[[CST_ONE:.*]] = arith.constant dense<1.000000e+00> : vector<8xf32>
// CHECK-COUNT-6:   select
// CHECK:           %[[VAL_79:.*]] = arith.select
// CHECK:           return %[[VAL_79]] : vector<8xf32>
// CHECK:         }
func.func @log1p_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.log1p %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}


// CHECK-LABEL:   func @tanh_scalar(
// CHECK-SAME:                      %[[VAL_0:.*]]: f32) -> f32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant -7.99881172 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant 7.99881172 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 4.000000e-04 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0.00489352457 : f32
// CHECK:           %[[VAL_5:.*]] = arith.constant 6.37261954E-4 : f32
// CHECK:           %[[VAL_6:.*]] = arith.constant 1.48572235E-5 : f32
// CHECK:           %[[VAL_7:.*]] = arith.constant 5.12229725E-8 : f32
// CHECK:           %[[VAL_8:.*]] = arith.constant -8.60467184E-11 : f32
// CHECK:           %[[VAL_9:.*]] = arith.constant 2.00018794E-13 : f32
// CHECK:           %[[VAL_10:.*]] = arith.constant -2.76076837E-16 : f32
// CHECK:           %[[VAL_11:.*]] = arith.constant 0.00489352504 : f32
// CHECK:           %[[VAL_12:.*]] = arith.constant 0.00226843474 : f32
// CHECK:           %[[VAL_13:.*]] = arith.constant 1.18534706E-4 : f32
// CHECK:           %[[VAL_14:.*]] = arith.constant 1.19825836E-6 : f32
// CHECK:           %[[VAL_15:.*]] = arith.cmpf ult, %[[VAL_0]], %[[VAL_2]] : f32
// CHECK:           %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_0]], %[[VAL_2]] : f32
// CHECK:           %[[VAL_17:.*]] = arith.cmpf ugt, %[[VAL_16]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_16]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_19:.*]] = math.absf %[[VAL_0]] : f32
// CHECK:           %[[VAL_20:.*]] = arith.cmpf olt, %[[VAL_19]], %[[VAL_3]] : f32
// CHECK:           %[[VAL_21:.*]] = arith.mulf %[[VAL_18]], %[[VAL_18]] : f32
// CHECK:           %[[VAL_22:.*]] = math.fma %[[VAL_21]], %[[VAL_10]], %[[VAL_9]] : f32
// CHECK:           %[[VAL_23:.*]] = math.fma %[[VAL_21]], %[[VAL_22]], %[[VAL_8]] : f32
// CHECK:           %[[VAL_24:.*]] = math.fma %[[VAL_21]], %[[VAL_23]], %[[VAL_7]] : f32
// CHECK:           %[[VAL_25:.*]] = math.fma %[[VAL_21]], %[[VAL_24]], %[[VAL_6]] : f32
// CHECK:           %[[VAL_26:.*]] = math.fma %[[VAL_21]], %[[VAL_25]], %[[VAL_5]] : f32
// CHECK:           %[[VAL_27:.*]] = math.fma %[[VAL_21]], %[[VAL_26]], %[[VAL_4]] : f32
// CHECK:           %[[VAL_28:.*]] = arith.mulf %[[VAL_18]], %[[VAL_27]] : f32
// CHECK:           %[[VAL_29:.*]] = math.fma %[[VAL_21]], %[[VAL_14]], %[[VAL_13]] : f32
// CHECK:           %[[VAL_30:.*]] = math.fma %[[VAL_21]], %[[VAL_29]], %[[VAL_12]] : f32
// CHECK:           %[[VAL_31:.*]] = math.fma %[[VAL_21]], %[[VAL_30]], %[[VAL_11]] : f32
// CHECK:           %[[VAL_32:.*]] = arith.divf %[[VAL_28]], %[[VAL_31]] : f32
// CHECK:           %[[VAL_33:.*]] = arith.select %[[VAL_20]], %[[VAL_18]], %[[VAL_32]] : f32
// CHECK:           return %[[VAL_33]] : f32
// CHECK:         }
func.func @tanh_scalar(%arg0: f32) -> f32 {
  %0 = math.tanh %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @tanh_vector(
// CHECK-SAME:                      %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<-7.99881172> : vector<8xf32>
// CHECK-NOT:       tanh
// CHECK-COUNT-2:   select
// CHECK:           %[[VAL_33:.*]] = arith.select
// CHECK:           return %[[VAL_33]] : vector<8xf32>
// CHECK:         }
func.func @tanh_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.tanh %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// We only approximate rsqrt for vectors and when the AVX2 option is enabled.
// CHECK-LABEL:   func @rsqrt_scalar
// AVX2-LABEL:    func @rsqrt_scalar
// CHECK:           math.rsqrt
// AVX2:            math.rsqrt
func.func @rsqrt_scalar(%arg0: f32) -> f32 {
  %0 = math.rsqrt %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @rsqrt_vector_8xf32
// CHECK:           math.rsqrt
// AVX2-LABEL:    func @rsqrt_vector_8xf32(
// AVX2-SAME:       %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// AVX2:   %[[VAL_1:.*]] = arith.constant dense<0x7F800000> : vector<8xf32>
// AVX2:   %[[VAL_2:.*]] = arith.constant dense<1.500000e+00> : vector<8xf32>
// AVX2:   %[[VAL_3:.*]] = arith.constant dense<-5.000000e-01> : vector<8xf32>
// AVX2:   %[[VAL_4:.*]] = arith.constant dense<1.17549435E-38> : vector<8xf32>
// AVX2:   %[[VAL_5:.*]] = arith.mulf %[[VAL_0]], %[[VAL_3]] : vector<8xf32>
// AVX2:   %[[VAL_6:.*]] = arith.cmpf olt, %[[VAL_0]], %[[VAL_4]] : vector<8xf32>
// AVX2:   %[[VAL_7:.*]] = arith.cmpf oeq, %[[VAL_0]], %[[VAL_1]] : vector<8xf32>
// AVX2:   %[[VAL_8:.*]] = arith.ori %[[VAL_6]], %[[VAL_7]] : vector<8xi1>
// AVX2:   %[[VAL_9:.*]] = x86vector.avx.rsqrt %[[VAL_0]] : vector<8xf32>
// AVX2:   %[[VAL_10:.*]] = arith.mulf %[[VAL_5]], %[[VAL_9]] : vector<8xf32>
// AVX2:   %[[VAL_11:.*]] = math.fma %[[VAL_9]], %[[VAL_10]], %[[VAL_2]] : vector<8xf32>
// AVX2:   %[[VAL_12:.*]] = arith.mulf %[[VAL_9]], %[[VAL_11]] : vector<8xf32>
// AVX2:   %[[VAL_13:.*]] = arith.select %[[VAL_8]], %[[VAL_9]], %[[VAL_12]] : vector<8xi1>, vector<8xf32>
// AVX2:   return %[[VAL_13]] : vector<8xf32>
// AVX2: }
func.func @rsqrt_vector_8xf32(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.rsqrt %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// Virtual vector width is not a multiple of an AVX2 vector width.
//
// CHECK-LABEL:  func @rsqrt_vector_5xf32
// CHECK:          math.rsqrt
// AVX2-LABEL:   func @rsqrt_vector_5xf32
// AVX2:           math.rsqrt
func.func @rsqrt_vector_5xf32(%arg0: vector<5xf32>) -> vector<5xf32> {
  %0 = math.rsqrt %arg0 : vector<5xf32>
  return %0 : vector<5xf32>
}

// One dimensional virtual vector expanded and unrolled into multiple AVX2-sized
// vectors.
//
// CHECK-LABEL: func @rsqrt_vector_16xf32
// CHECK:         math.rsqrt
// AVX2-LABEL:  func @rsqrt_vector_16xf32(
// AVX2-SAME:     %[[ARG:.*]]: vector<16xf32>
// AVX2-SAME:   ) -> vector<16xf32>
// AVX2:          %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<2x8xf32>
// AVX2:          %[[EXPAND:.*]] = vector.shape_cast %[[ARG]] : vector<16xf32> to vector<2x8xf32>
// AVX2:          %[[VEC0:.*]] = vector.extract %[[EXPAND]][0]
// AVX2:          %[[RSQRT0:.*]] = x86vector.avx.rsqrt %[[VEC0]]
// AVX2:          %[[VEC1:.*]] = vector.extract %[[EXPAND]][1]
// AVX2:          %[[RSQRT1:.*]] = x86vector.avx.rsqrt %[[VEC1]]
// AVX2:          %[[RESULT0:.*]] = vector.insert %[[RSQRT0]], %[[INIT]] [0]
// AVX2:          %[[RESULT1:.*]] = vector.insert %[[RSQRT1]], %[[RESULT0]] [1]
// AVX2:          %[[RSQRT:.*]] = vector.shape_cast %[[RESULT1]] : vector<2x8xf32> to vector<16xf32>
func.func @rsqrt_vector_16xf32(%arg0: vector<16xf32>) -> vector<16xf32> {
  %0 = math.rsqrt %arg0 : vector<16xf32>
  return %0 : vector<16xf32>
}

// Two dimensional virtual vector unrolled into multiple AVX2-sized vectors.
//
// CHECK-LABEL: func @rsqrt_vector_2x8xf32
// CHECK:         math.rsqrt
// AVX2-LABEL:  func @rsqrt_vector_2x8xf32(
// AVX2-SAME:     %[[ARG:.*]]: vector<2x8xf32>
// AVX2-SAME:   ) -> vector<2x8xf32>
// AVX2:          %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<2x8xf32>
// AVX2-NOT:      vector.shape_cast
// AVX2:          %[[VEC0:.*]] = vector.extract %[[ARG]][0]
// AVX2:          %[[RSQRT0:.*]] = x86vector.avx.rsqrt %[[VEC0]]
// AVX2:          %[[VEC1:.*]] = vector.extract %[[ARG]][1]
// AVX2:          %[[RSQRT1:.*]] = x86vector.avx.rsqrt %[[VEC1]]
// AVX2:          %[[RESULT0:.*]] = vector.insert %[[RSQRT0]], %[[INIT]] [0]
// AVX2:          %[[RESULT1:.*]] = vector.insert %[[RSQRT1]], %[[RESULT0]] [1]
// AVX2-NOT:      vector.shape_cast
func.func @rsqrt_vector_2x8xf32(%arg0: vector<2x8xf32>) -> vector<2x8xf32> {
  %0 = math.rsqrt %arg0 : vector<2x8xf32>
  return %0 : vector<2x8xf32>
}

// Two dimensional virtual vector expanded and unrolled into multiple AVX2-sized
// vectors.
//
// CHECK-LABEL: func @rsqrt_vector_2x16xf32
// CHECK:         math.rsqrt
// AVX2-LABEL:  func @rsqrt_vector_2x16xf32(
// AVX2-SAME:     %[[ARG:.*]]: vector<2x16xf32>
// AVX2-SAME:   ) -> vector<2x16xf32>
// AVX2:          %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<2x2x8xf32>
// AVX2:          %[[EXPAND:.*]] = vector.shape_cast %[[ARG]] : vector<2x16xf32> to vector<2x2x8xf32>
// AVX2:          %[[VEC00:.*]] = vector.extract %[[EXPAND]][0, 0]
// AVX2:          %[[RSQRT00:.*]] = x86vector.avx.rsqrt %[[VEC00]]
// AVX2:          %[[VEC01:.*]] = vector.extract %[[EXPAND]][0, 1]
// AVX2:          %[[RSQRT01:.*]] = x86vector.avx.rsqrt %[[VEC01]]
// AVX2:          %[[VEC10:.*]] = vector.extract %[[EXPAND]][1, 0]
// AVX2:          %[[RSQRT10:.*]] = x86vector.avx.rsqrt %[[VEC10]]
// AVX2:          %[[VEC11:.*]] = vector.extract %[[EXPAND]][1, 1]
// AVX2:          %[[RSQRT11:.*]] = x86vector.avx.rsqrt %[[VEC11]]
// AVX2:          %[[RESULT0:.*]] = vector.insert %[[RSQRT00]], %[[INIT]] [0, 0]
// AVX2:          %[[RESULT1:.*]] = vector.insert %[[RSQRT01]], %[[RESULT0]] [0, 1]
// AVX2:          %[[RESULT2:.*]] = vector.insert %[[RSQRT10]], %[[RESULT1]] [1, 0]
// AVX2:          %[[RESULT3:.*]] = vector.insert %[[RSQRT11]], %[[RESULT2]] [1, 1]
// AVX2:          %[[RSQRT:.*]] = vector.shape_cast %[[RESULT3]] : vector<2x2x8xf32> to vector<2x16xf32>
func.func @rsqrt_vector_2x16xf32(%arg0: vector<2x16xf32>) -> vector<2x16xf32> {
  %0 = math.rsqrt %arg0 : vector<2x16xf32>
  return %0 : vector<2x16xf32>
}

// CHECK-LABEL: @atan_scalar
// CHECK-DAG:  %[[ONE:.+]] = arith.constant 1.000000e+00
// CHECK-DAG:  %[[N1:.+]] = arith.constant 0.144182831
// CHECK-DAG:  %[[N2:.+]] = arith.constant -0.349992335
// CHECK-DAG:  %[[N3:.+]] = arith.constant -0.0106783099
// CHECK-DAG:  %[[N4:.+]] = arith.constant 1.00209987
// CHECK-DAG:  %[[HALF_PI:.+]] = arith.constant 1.57079637
// CHECK-DAG:  %[[ABS:.+]] = math.absf %arg0
// CHECK-DAG:  %[[DIV:.+]] = arith.divf %cst, %[[ABS]]
// CHECK-DAG:  %[[CMP:.+]] = arith.cmpf olt, %[[ABS]], %[[DIV]]
// CHECK-DAG:  %[[SEL:.+]] = arith.select %[[CMP]], %[[ABS]], %[[DIV]]
// CHECK-DAG:  %[[P0:.+]] = math.fma %[[SEL]], %[[N1]], %[[N2]]
// CHECK-DAG:  %[[P1:.+]] = math.fma %[[SEL]], %[[P0]], %[[N3]]
// CHECK-DAG:  %[[P2:.+]] = math.fma %[[SEL]], %[[P1]], %[[N4]]
// CHECK-DAG:  %[[P3:.+]] = arith.mulf %[[SEL]], %[[P2]]
// CHECK-DAG:  %[[SUB:.+]] = arith.subf %[[HALF_PI]], %[[P3]]
// CHECK-DAG:  %[[EST:.+]] = arith.select %[[CMP]], %[[P3]], %[[SUB]]
// CHECK-DAG:  %[[RES:.+]] = math.copysign %[[EST]], %arg0
// CHECK:  return %[[RES]]
func.func @atan_scalar(%arg0: f32) -> f32 {
  %0 = math.atan %arg0 : f32
  return %0 : f32
}


// CHECK-LABEL: @atan2_scalar

// ATan approximation:
// CHECK-DAG:  %[[ONE:.+]] = arith.constant 1.000000e+00
// CHECK-DAG:  %[[N1:.+]] = arith.constant 0.144182831
// CHECK-DAG:  %[[N2:.+]] = arith.constant -0.349992335
// CHECK-DAG:  %[[N3:.+]] = arith.constant -0.0106783099
// CHECK-DAG:  %[[N4:.+]] = arith.constant 1.00209987
// CHECK-DAG:  %[[HALF_PI:.+]] = arith.constant 1.57079637
// CHECK-DAG:  %[[ARG0:.+]] = arith.extf %arg0 : f16 to f32
// CHECK-DAG:  %[[ARG1:.+]] = arith.extf %arg1 : f16 to f32
// CHECK-DAG:  %[[RATIO:.+]] = arith.divf %[[ARG0]], %[[ARG1]]
// CHECK-DAG:  %[[ABS:.+]] = math.absf %[[RATIO]]
// CHECK-DAG:  %[[DIV:.+]] = arith.divf %cst, %[[ABS]]
// CHECK-DAG:  %[[CMP:.+]] = arith.cmpf olt, %[[ABS]], %[[DIV]]
// CHECK-DAG:  %[[SEL:.+]] = arith.select %[[CMP]], %[[ABS]], %[[DIV]]
// CHECK-DAG:  %[[P0:.+]] = math.fma %[[SEL]], %[[N1]], %[[N2]]
// CHECK-DAG:  %[[P1:.+]] = math.fma %[[SEL]], %[[P0]], %[[N3]]
// CHECK-DAG:  %[[P2:.+]] = math.fma %[[SEL]], %[[P1]], %[[N4]]
// CHECK-DAG:  %[[P3:.+]] = arith.mulf %[[SEL]], %[[P2]]
// CHECK-DAG:  %[[SUB:.+]] = arith.subf %[[HALF_PI]], %[[P3]]
// CHECK-DAG:  %[[EST:.+]] = arith.select %[[CMP]], %[[P3]], %[[SUB]]
// CHECK-DAG:  %[[ATAN:.+]] = math.copysign %[[EST]], %[[RATIO]]

// Handle the case of x < 0:
// CHECK-DAG: %[[ZERO:.+]] = arith.constant 0.000000e+00
// CHECK-DAG: %[[PI:.+]] = arith.constant 3.14159274
// CHECK-DAG: %[[ADD_PI:.+]] = arith.addf %[[ATAN]], %[[PI]]
// CHECK-DAG: %[[SUB_PI:.+]] = arith.subf %[[ATAN]], %[[PI]]
// CHECK-DAG: %[[CMP_ATAN:.+]] = arith.cmpf ogt, %[[ATAN]], %[[ZERO]]
// CHECK-DAG: %[[ATAN_ADJUST:.+]] = arith.select %[[CMP_ATAN]], %[[SUB_PI]], %[[ADD_PI]]
// CHECK-DAG: %[[X_NEG:.+]] = arith.cmpf ogt, %[[ARG1]], %[[ZERO]]
// CHECK-DAG: %[[ATAN_EST:.+]] = arith.select %[[X_NEG]], %[[ATAN]], %[[ATAN_ADJUST]]

// Handle PI / 2 edge case:
// CHECK-DAG: %[[X_ZERO:.+]] = arith.cmpf oeq, %[[ARG1]], %[[ZERO]]
// CHECK-DAG: %[[Y_POS:.+]] = arith.cmpf ogt, %[[ARG0]], %[[ZERO]]
// CHECK-DAG: %[[IS_HALF_PI:.+]] = arith.andi %[[X_ZERO]], %[[Y_POS]]
// CHECK-DAG: %[[EDGE1:.+]] = arith.select %[[IS_HALF_PI]], %[[HALF_PI]], %[[ATAN_EST]]

// Handle -PI / 2 edge case:
// CHECK-DAG: %[[NEG_HALF_PI:.+]] = arith.constant -1.57079637
// CHECK-DAG: %[[Y_NEG:.+]] = arith.cmpf olt, %[[ARG0]], %[[ZERO]]
// CHECK-DAG: %[[IS_NEG_HALF_PI:.+]] = arith.andi %[[X_ZERO]], %[[Y_NEG]]
// CHECK-DAG: %[[EDGE2:.+]] = arith.select %[[IS_NEG_HALF_PI]], %[[NEG_HALF_PI]], %[[EDGE1]]

// Handle Nan edgecase:
// CHECK-DAG: %[[Y_ZERO:.+]] = arith.cmpf oeq, %[[ARG0]], %[[ZERO]]
// CHECK-DAG: %[[X_Y_ZERO:.+]] = arith.andi %[[X_ZERO]], %[[Y_ZERO]]
// CHECK-DAG: %[[NAN:.+]] = arith.constant 0x7FC00000
// CHECK-DAG: %[[EDGE3:.+]] = arith.select %[[X_Y_ZERO]], %[[NAN]], %[[EDGE2]]
// CHECK: %[[RET:.+]] = arith.truncf %[[EDGE3]]
// CHECK: return %[[RET]]

func.func @atan2_scalar(%arg0: f16, %arg1: f16) -> f16 {
  %0 = math.atan2 %arg0, %arg1 : f16
  return %0 : f16
}
