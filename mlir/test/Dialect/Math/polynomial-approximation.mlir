// RUN: mlir-opt %s -test-math-polynomial-approximation | FileCheck %s

// Check that all math functions lowered to approximations built from
// standard operations (add, mul, fma, shift, etc...).

// CHECK-LABEL:   func @exp_scalar(
// CHECK-SAME:                     %[[VAL_0:.*]]: f32) -> f32 {
// CHECK:           %[[VAL_1:.*]] = constant 0.693147182 : f32
// CHECK:           %[[VAL_2:.*]] = constant 1.44269502 : f32
// CHECK:           %[[VAL_3:.*]] = constant 1.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = constant 0.499705136 : f32
// CHECK:           %[[VAL_5:.*]] = constant 0.168738902 : f32
// CHECK:           %[[VAL_6:.*]] = constant 0.0366896503 : f32
// CHECK:           %[[VAL_7:.*]] = constant 1.314350e-02 : f32
// CHECK:           %[[VAL_8:.*]] = constant 23 : i32
// CHECK:           %[[VAL_9:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_10:.*]] = constant 0x7F800000 : f32
// CHECK:           %[[VAL_11:.*]] = constant 0xFF800000 : f32
// CHECK:           %[[VAL_12:.*]] = constant 1.17549435E-38 : f32
// CHECK:           %[[VAL_13:.*]] = constant 127 : i32
// CHECK:           %[[VAL_14:.*]] = constant -127 : i32
// CHECK:           %[[VAL_15:.*]] = mulf %[[VAL_0]], %[[VAL_2]] : f32
// CHECK:           %[[VAL_16:.*]] = floorf %[[VAL_15]] : f32
// CHECK:           %[[VAL_17:.*]] = mulf %[[VAL_16]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_18:.*]] = subf %[[VAL_0]], %[[VAL_17]] : f32
// CHECK:           %[[VAL_19:.*]] = mulf %[[VAL_18]], %[[VAL_18]] : f32
// CHECK:           %[[VAL_20:.*]] = mulf %[[VAL_19]], %[[VAL_19]] : f32
// CHECK:           %[[VAL_21:.*]] = fmaf %[[VAL_3]], %[[VAL_18]], %[[VAL_3]] : f32
// CHECK:           %[[VAL_22:.*]] = fmaf %[[VAL_5]], %[[VAL_18]], %[[VAL_4]] : f32
// CHECK:           %[[VAL_23:.*]] = fmaf %[[VAL_7]], %[[VAL_18]], %[[VAL_6]] : f32
// CHECK:           %[[VAL_24:.*]] = fmaf %[[VAL_22]], %[[VAL_19]], %[[VAL_21]] : f32
// CHECK:           %[[VAL_25:.*]] = fmaf %[[VAL_23]], %[[VAL_20]], %[[VAL_24]] : f32
// CHECK:           %[[VAL_26:.*]] = fptosi %[[VAL_16]] : f32 to i32
// CHECK:           %[[VAL_27:.*]] = addi %[[VAL_26]], %[[VAL_13]] : i32
// CHECK:           %[[VAL_28:.*]] = shift_left %[[VAL_27]], %[[VAL_8]] : i32
// CHECK:           %[[VAL_29:.*]] = llvm.bitcast %[[VAL_28]] : i32 to f32
// CHECK:           %[[VAL_30:.*]] = mulf %[[VAL_25]], %[[VAL_29]] : f32
// CHECK:           %[[VAL_31:.*]] = cmpi sle, %[[VAL_26]], %[[VAL_13]] : i32
// CHECK:           %[[VAL_32:.*]] = cmpi sge, %[[VAL_26]], %[[VAL_14]] : i32
// CHECK:           %[[VAL_33:.*]] = cmpf oeq, %[[VAL_0]], %[[VAL_11]] : f32
// CHECK:           %[[VAL_34:.*]] = cmpf ogt, %[[VAL_0]], %[[VAL_9]] : f32
// CHECK:           %[[VAL_35:.*]] = and %[[VAL_31]], %[[VAL_32]] : i1
// CHECK:           %[[VAL_36:.*]] = select %[[VAL_33]], %[[VAL_9]], %[[VAL_12]] : f32
// CHECK:           %[[VAL_37:.*]] = select %[[VAL_34]], %[[VAL_10]], %[[VAL_36]] : f32
// CHECK:           %[[VAL_38:.*]] = select %[[VAL_35]], %[[VAL_30]], %[[VAL_37]] : f32
// CHECK:           return %[[VAL_38]] : f32
// CHECK:         }
func @exp_scalar(%arg0: f32) -> f32 {
  %0 = math.exp %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @exp_vector(
// CHECK-SAME:                     %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:           %[[VAL_1:.*]] = constant dense<0.693147182> : vector<8xf32>
// CHECK-NOT:       exp
// CHECK-COUNT-2:   select
// CHECK:           %[[VAL_38:.*]] = select
// CHECK:           return %[[VAL_38]] : vector<8xf32>
// CHECK:         }
func @exp_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.exp %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL:   func @expm1_scalar(
// CHECK-SAME:                       %[[X:.*]]: f32) -> f32 {
// CHECK:           %[[CST_MINUSONE:.*]] = constant -1.000000e+00 : f32
// CHECK:           %[[CST_LOG2E:.*]] = constant 1.44269502 : f32
// CHECK:           %[[CST_ONE:.*]] = constant 1.000000e+00 : f32
// CHECK:           %[[BEGIN_EXP_X:.*]] = mulf %[[X]], %[[CST_LOG2E]] : f32
// CHECK-NOT:       exp
// CHECK-COUNT-2:   select
// CHECK:           %[[EXP_X:.*]] = select
// CHECK:           %[[VAL_58:.*]] = cmpf oeq, %[[EXP_X]], %[[CST_ONE]] : f32
// CHECK:           %[[VAL_59:.*]] = subf %[[EXP_X]], %[[CST_ONE]] : f32
// CHECK:           %[[VAL_60:.*]] = cmpf oeq, %[[VAL_59]], %[[CST_MINUSONE]] : f32
// CHECK-NOT:       log
// CHECK-COUNT-5:   select
// CHECK:           %[[LOG_U:.*]] = select
// CHECK:           %[[VAL_104:.*]] = cmpf oeq, %[[LOG_U]], %[[EXP_X]] : f32
// CHECK:           %[[VAL_105:.*]] = divf %[[X]], %[[LOG_U]] : f32
// CHECK:           %[[VAL_106:.*]] = mulf %[[VAL_59]], %[[VAL_105]] : f32
// CHECK:           %[[VAL_107:.*]] = select %[[VAL_104]], %[[EXP_X]], %[[VAL_106]] : f32
// CHECK:           %[[VAL_108:.*]] = select %[[VAL_60]], %[[CST_MINUSONE]], %[[VAL_107]] : f32
// CHECK:           %[[VAL_109:.*]] = select %[[VAL_58]], %[[X]], %[[VAL_108]] : f32
// CHECK:           return %[[VAL_109]] : f32
// CHECK:         }
func @expm1_scalar(%arg0: f32) -> f32 {
  %0 = math.expm1 %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @expm1_vector(
// CHECK-SAME:                       %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:           %[[VAL_1:.*]] = constant dense<-1.000000e+00> : vector<8xf32>
// CHECK-NOT:       exp
// CHECK-COUNT-3:   select
// CHECK-NOT:       log
// CHECK-COUNT-6:   vector.broadcast
// CHECK-COUNT-5:   select
// CHECK-NOT:       expm1
// CHECK-COUNT-2:   select
// CHECK:           %[[VAL_115:.*]] = select
// CHECK:           return %[[VAL_115]] : vector<8xf32>
// CHECK:         }
func @expm1_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.expm1 %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL:   func @log_scalar(
// CHECK-SAME:                     %[[X:.*]]: f32) -> f32 {
// CHECK:           %[[VAL_1:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = constant 1.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = constant -5.000000e-01 : f32
// CHECK:           %[[VAL_4:.*]] = constant 8388608 : i32
// CHECK:           %[[VAL_5:.*]] = constant -8388608 : i32
// CHECK:           %[[VAL_6:.*]] = constant 2139095040 : i32
// CHECK:           %[[VAL_7:.*]] = constant 2143289344 : i32
// CHECK:           %[[VAL_8:.*]] = constant 0.707106769 : f32
// CHECK:           %[[VAL_9:.*]] = constant 0.0703768358 : f32
// CHECK:           %[[VAL_10:.*]] = constant -0.115146101 : f32
// CHECK:           %[[VAL_11:.*]] = constant 0.116769984 : f32
// CHECK:           %[[VAL_12:.*]] = constant -0.12420141 : f32
// CHECK:           %[[VAL_13:.*]] = constant 0.142493233 : f32
// CHECK:           %[[VAL_14:.*]] = constant -0.166680574 : f32
// CHECK:           %[[VAL_15:.*]] = constant 0.200007141 : f32
// CHECK:           %[[VAL_16:.*]] = constant -0.24999994 : f32
// CHECK:           %[[VAL_17:.*]] = constant 0.333333313 : f32
// CHECK:           %[[VAL_18:.*]] = constant 1.260000e+02 : f32
// CHECK:           %[[VAL_19:.*]] = constant 5.000000e-01 : f32
// CHECK:           %[[VAL_20:.*]] = constant -2139095041 : i32
// CHECK:           %[[VAL_21:.*]] = constant 23 : i32
// CHECK:           %[[CST_LN2:.*]] = constant 0.693147182 : f32
// CHECK:           %[[VAL_23:.*]] = llvm.bitcast %[[VAL_4]] : i32 to f32
// CHECK:           %[[VAL_24:.*]] = llvm.bitcast %[[VAL_5]] : i32 to f32
// CHECK:           %[[VAL_25:.*]] = llvm.bitcast %[[VAL_6]] : i32 to f32
// CHECK:           %[[VAL_26:.*]] = llvm.bitcast %[[VAL_7]] : i32 to f32
// CHECK:           %[[VAL_27:.*]] = cmpf ogt, %[[X]], %[[VAL_23]] : f32
// CHECK:           %[[VAL_28:.*]] = select %[[VAL_27]], %[[X]], %[[VAL_23]] : f32
// CHECK-NOT:       frexp
// CHECK:           %[[VAL_29:.*]] = llvm.bitcast %[[VAL_20]] : i32 to f32
// CHECK:           %[[VAL_30:.*]] = llvm.bitcast %[[VAL_19]] : f32 to i32
// CHECK:           %[[VAL_31:.*]] = llvm.bitcast %[[VAL_29]] : f32 to i32
// CHECK:           %[[VAL_32:.*]] = llvm.bitcast %[[VAL_28]] : f32 to i32
// CHECK:           %[[VAL_33:.*]] = llvm.and %[[VAL_32]], %[[VAL_31]]  : i32
// CHECK:           %[[VAL_34:.*]] = llvm.or %[[VAL_33]], %[[VAL_30]]  : i32
// CHECK:           %[[VAL_35:.*]] = llvm.bitcast %[[VAL_34]] : i32 to f32
// CHECK:           %[[VAL_36:.*]] = llvm.bitcast %[[VAL_28]] : f32 to i32
// CHECK:           %[[VAL_37:.*]] = shift_right_unsigned %[[VAL_36]], %[[VAL_21]] : i32
// CHECK:           %[[FREXP_X:.*]] = sitofp %[[VAL_37]] : i32 to f32
// CHECK:           %[[VAL_39:.*]] = subf %[[FREXP_X]], %[[VAL_18]] : f32
// CHECK:           %[[VAL_40:.*]] = cmpf olt, %[[VAL_35]], %[[VAL_8]] : f32
// CHECK:           %[[VAL_41:.*]] = select %[[VAL_40]], %[[VAL_35]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_42:.*]] = subf %[[VAL_35]], %[[VAL_2]] : f32
// CHECK:           %[[VAL_43:.*]] = select %[[VAL_40]], %[[VAL_2]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_44:.*]] = subf %[[VAL_39]], %[[VAL_43]] : f32
// CHECK:           %[[VAL_45:.*]] = addf %[[VAL_42]], %[[VAL_41]] : f32
// CHECK:           %[[VAL_46:.*]] = mulf %[[VAL_45]], %[[VAL_45]] : f32
// CHECK:           %[[VAL_47:.*]] = mulf %[[VAL_46]], %[[VAL_45]] : f32
// CHECK:           %[[VAL_48:.*]] = fmaf %[[VAL_9]], %[[VAL_45]], %[[VAL_10]] : f32
// CHECK:           %[[VAL_49:.*]] = fmaf %[[VAL_12]], %[[VAL_45]], %[[VAL_13]] : f32
// CHECK:           %[[VAL_50:.*]] = fmaf %[[VAL_15]], %[[VAL_45]], %[[VAL_16]] : f32
// CHECK:           %[[VAL_51:.*]] = fmaf %[[VAL_48]], %[[VAL_45]], %[[VAL_11]] : f32
// CHECK:           %[[VAL_52:.*]] = fmaf %[[VAL_49]], %[[VAL_45]], %[[VAL_14]] : f32
// CHECK:           %[[VAL_53:.*]] = fmaf %[[VAL_50]], %[[VAL_45]], %[[VAL_17]] : f32
// CHECK:           %[[VAL_54:.*]] = fmaf %[[VAL_51]], %[[VAL_47]], %[[VAL_52]] : f32
// CHECK:           %[[VAL_55:.*]] = fmaf %[[VAL_54]], %[[VAL_47]], %[[VAL_53]] : f32
// CHECK:           %[[VAL_56:.*]] = mulf %[[VAL_55]], %[[VAL_47]] : f32
// CHECK:           %[[VAL_57:.*]] = fmaf %[[VAL_3]], %[[VAL_46]], %[[VAL_56]] : f32
// CHECK:           %[[VAL_58:.*]] = addf %[[VAL_45]], %[[VAL_57]] : f32
// CHECK:           %[[VAL_59:.*]] = fmaf %[[VAL_44]], %[[CST_LN2]], %[[VAL_58]] : f32
// CHECK:           %[[VAL_60:.*]] = cmpf ult, %[[X]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_61:.*]] = cmpf oeq, %[[X]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_62:.*]] = cmpf oeq, %[[X]], %[[VAL_25]] : f32
// CHECK:           %[[VAL_63:.*]] = select %[[VAL_62]], %[[VAL_25]], %[[VAL_59]] : f32
// CHECK:           %[[VAL_64:.*]] = select %[[VAL_60]], %[[VAL_26]], %[[VAL_63]] : f32
// CHECK:           %[[VAL_65:.*]] = select %[[VAL_61]], %[[VAL_24]], %[[VAL_64]] : f32
// CHECK:           return %[[VAL_65]] : f32
// CHECK:         }
func @log_scalar(%arg0: f32) -> f32 {
  %0 = math.log %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @log_vector(
// CHECK-SAME:                     %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:           %[[CST_LN2:.*]] = constant dense<0.693147182> : vector<8xf32>
// CHECK-COUNT-6:   vector.broadcast
// CHECK-COUNT-4:   select
// CHECK:           %[[VAL_71:.*]] = select
// CHECK:           return %[[VAL_71]] : vector<8xf32>
// CHECK:         }
func @log_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.log %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL:   func @log2_scalar(
// CHECK-SAME:                      %[[VAL_0:.*]]: f32) -> f32 {
// CHECK:           %[[CST_LOG2E:.*]] = constant 1.44269502 : f32
// CHECK-COUNT-5:   select
// CHECK:           %[[VAL_65:.*]] = select
// CHECK:           return %[[VAL_65]] : f32
// CHECK:         }
func @log2_scalar(%arg0: f32) -> f32 {
  %0 = math.log2 %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @log2_vector(
// CHECK-SAME:                      %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:           %[[CST_LOG2E:.*]] = constant dense<1.44269502> : vector<8xf32>
// CHECK-COUNT-6:   vector.broadcast
// CHECK-COUNT-4:   select
// CHECK:           %[[VAL_71:.*]] = select
// CHECK:           return %[[VAL_71]] : vector<8xf32>
// CHECK:         }
func @log2_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.log2 %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL:   func @log1p_scalar(
// CHECK-SAME:                       %[[X:.*]]: f32) -> f32 {
// CHECK:           %[[CST_ONE:.*]] = constant 1.000000e+00 : f32
// CHECK:           %[[U:.*]] = addf %[[X]], %[[CST_ONE]] : f32
// CHECK:           %[[U_SMALL:.*]] = cmpf oeq, %[[U]], %[[CST_ONE]] : f32
// CHECK-NOT:       log
// CHECK-COUNT-5:   select
// CHECK:           %[[LOG_U:.*]] = select
// CHECK:           %[[U_INF:.*]] = cmpf oeq, %[[U]], %[[LOG_U]] : f32
// CHECK:           %[[VAL_69:.*]] = subf %[[U]], %[[CST_ONE]] : f32
// CHECK:           %[[VAL_70:.*]] = divf %[[LOG_U]], %[[VAL_69]] : f32
// CHECK:           %[[LOG_LARGE:.*]] = mulf %[[X]], %[[VAL_70]] : f32
// CHECK:           %[[VAL_72:.*]] = llvm.or %[[U_SMALL]], %[[U_INF]]  : i1
// CHECK:           %[[APPROX:.*]] = select %[[VAL_72]], %[[X]], %[[LOG_LARGE]] : f32
// CHECK:           return %[[APPROX]] : f32
// CHECK:         }
func @log1p_scalar(%arg0: f32) -> f32 {
  %0 = math.log1p %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @log1p_vector(
// CHECK-SAME:                       %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:           %[[CST_ONE:.*]] = constant dense<1.000000e+00> : vector<8xf32>
// CHECK-COUNT-6:   vector.broadcast
// CHECK-COUNT-5:   select
// CHECK:           %[[VAL_79:.*]] = select
// CHECK:           return %[[VAL_79]] : vector<8xf32>
// CHECK:         }
func @log1p_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.log1p %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}


// CHECK-LABEL:   func @tanh_scalar(
// CHECK-SAME:                      %[[VAL_0:.*]]: f32) -> f32 {
// CHECK:           %[[VAL_1:.*]] = constant -7.90531111 : f32
// CHECK:           %[[VAL_2:.*]] = constant 7.90531111 : f32
// CHECK:           %[[VAL_3:.*]] = constant 4.000000e-04 : f32
// CHECK:           %[[VAL_4:.*]] = constant 0.00489352457 : f32
// CHECK:           %[[VAL_5:.*]] = constant 6.37261954E-4 : f32
// CHECK:           %[[VAL_6:.*]] = constant 1.48572235E-5 : f32
// CHECK:           %[[VAL_7:.*]] = constant 5.12229725E-8 : f32
// CHECK:           %[[VAL_8:.*]] = constant -8.60467184E-11 : f32
// CHECK:           %[[VAL_9:.*]] = constant 2.00018794E-13 : f32
// CHECK:           %[[VAL_10:.*]] = constant -2.76076837E-16 : f32
// CHECK:           %[[VAL_11:.*]] = constant 0.00489352504 : f32
// CHECK:           %[[VAL_12:.*]] = constant 0.00226843474 : f32
// CHECK:           %[[VAL_13:.*]] = constant 1.18534706E-4 : f32
// CHECK:           %[[VAL_14:.*]] = constant 1.19825836E-6 : f32
// CHECK:           %[[VAL_15:.*]] = cmpf olt, %[[VAL_0]], %[[VAL_2]] : f32
// CHECK:           %[[VAL_16:.*]] = select %[[VAL_15]], %[[VAL_0]], %[[VAL_2]] : f32
// CHECK:           %[[VAL_17:.*]] = cmpf ogt, %[[VAL_16]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_18:.*]] = select %[[VAL_17]], %[[VAL_16]], %[[VAL_1]] : f32
// CHECK:           %[[VAL_19:.*]] = absf %[[VAL_0]] : f32
// CHECK:           %[[VAL_20:.*]] = cmpf olt, %[[VAL_19]], %[[VAL_3]] : f32
// CHECK:           %[[VAL_21:.*]] = mulf %[[VAL_18]], %[[VAL_18]] : f32
// CHECK:           %[[VAL_22:.*]] = fmaf %[[VAL_21]], %[[VAL_10]], %[[VAL_9]] : f32
// CHECK:           %[[VAL_23:.*]] = fmaf %[[VAL_21]], %[[VAL_22]], %[[VAL_8]] : f32
// CHECK:           %[[VAL_24:.*]] = fmaf %[[VAL_21]], %[[VAL_23]], %[[VAL_7]] : f32
// CHECK:           %[[VAL_25:.*]] = fmaf %[[VAL_21]], %[[VAL_24]], %[[VAL_6]] : f32
// CHECK:           %[[VAL_26:.*]] = fmaf %[[VAL_21]], %[[VAL_25]], %[[VAL_5]] : f32
// CHECK:           %[[VAL_27:.*]] = fmaf %[[VAL_21]], %[[VAL_26]], %[[VAL_4]] : f32
// CHECK:           %[[VAL_28:.*]] = mulf %[[VAL_18]], %[[VAL_27]] : f32
// CHECK:           %[[VAL_29:.*]] = fmaf %[[VAL_21]], %[[VAL_14]], %[[VAL_13]] : f32
// CHECK:           %[[VAL_30:.*]] = fmaf %[[VAL_21]], %[[VAL_29]], %[[VAL_12]] : f32
// CHECK:           %[[VAL_31:.*]] = fmaf %[[VAL_21]], %[[VAL_30]], %[[VAL_11]] : f32
// CHECK:           %[[VAL_32:.*]] = divf %[[VAL_28]], %[[VAL_31]] : f32
// CHECK:           %[[VAL_33:.*]] = select %[[VAL_20]], %[[VAL_18]], %[[VAL_32]] : f32
// CHECK:           return %[[VAL_33]] : f32
// CHECK:         }
func @tanh_scalar(%arg0: f32) -> f32 {
  %0 = math.tanh %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL:   func @tanh_vector(
// CHECK-SAME:                      %[[VAL_0:.*]]: vector<8xf32>) -> vector<8xf32> {
// CHECK:           %[[VAL_1:.*]] = constant dense<-7.90531111> : vector<8xf32>
// CHECK-NOT:       tanh
// CHECK-COUNT-2:   select
// CHECK:           %[[VAL_33:.*]] = select
// CHECK:           return %[[VAL_33]] : vector<8xf32>
// CHECK:         }
func @tanh_vector(%arg0: vector<8xf32>) -> vector<8xf32> {
  %0 = math.tanh %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}
