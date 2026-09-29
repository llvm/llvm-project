; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -verify-machineinstrs | FileCheck %s
;
; Test that foldMemoryOperandImpl() doesn't drop subreg / read-undef flags.


define void @fun_llvm_stress_reduced(ptr, ptr, ptr, i64, i8) {
; CHECK: .text
BB:
  %A4 = alloca <4 x i64>
  %A1 = alloca <8 x i1>
  %E6 = extractelement <4 x i1> undef, i32 3
  %L23 = load i8, ptr %0
  %B27 = fmul double 0x59A989483BA7E0C6, undef
  %L30 = load i16, ptr undef
  store i16 -11933, ptr undef
  %L46 = load i16, ptr undef
  %L61 = load i16, ptr undef
  %Sl74 = select i1 undef, i1 undef, i1 true
  br label %CF846

CF846:                                            ; preds = %CF877, %BB
  %I86 = insertelement <4 x i1> undef, i1 undef, i32 0
  %Cmp89 = icmp ne i64 undef, %3
  %L90 = load i16, ptr undef
  %Shuff92 = shufflevector <4 x i16> zeroinitializer, <4 x i16> zeroinitializer, <4 x i32> <i32 0, i32 2, i32 undef, i32 6>
  br label %CF877

CF877:                                            ; preds = %CF846
  store i16 %L61, ptr undef
  %Cmp110 = icmp eq i16 %L61, undef
  br i1 %Cmp110, label %CF846, label %CF862

CF862:                                            ; preds = %CF877
  %I114 = insertelement <4 x i64> zeroinitializer, i64 0, i32 0
  %B115 = shl <4 x i64> zeroinitializer, %I114
  %Sl124 = select i1 true, ptr %A1, ptr %A1
  %B130 = frem double %B27, 0x59A989483BA7E0C6
  %E143 = extractelement <4 x i64> %B115, i32 1
  %Sl148 = select i1 %Cmp89, <1 x i32> undef, <1 x i32> zeroinitializer
  br label %CF855

CF855:                                            ; preds = %CF855, %CF862
  %Sl171 = select i1 %Sl74, i1 %E6, i1 undef
  br i1 %Sl171, label %CF855, label %CF874

CF874:                                            ; preds = %CF855
  %L196 = load i16, ptr undef
  %B207 = or i8 %4, %L23
  %L211 = load <8 x i1>, ptr %Sl124
  %B215 = fdiv double 0x8421A9C0D21F6D3E, %B130
  %L218 = load i16, ptr %1
  %Sl223 = select i1 %Sl171, <4 x i1> %I86, <4 x i1> undef
  br label %CF826

CF826:                                            ; preds = %CF866, %CF910, %CF874
  %B245 = ashr i16 -11933, %L46
  br label %CF910

CF910:                                            ; preds = %CF826
  %L257 = load i8, ptr %0
  %BC262 = bitcast i64 %E143 to double
  store i16 %L196, ptr %1
  %E266 = extractelement <4 x i16> %Shuff92, i32 0
  %Sl271 = select i1 %Cmp89, i1 %Cmp89, i1 %Cmp110
  br i1 %Sl271, label %CF826, label %CF866

CF866:                                            ; preds = %CF910
  store i64 %E143, ptr %2
  %I276 = insertelement <4 x double> undef, double %BC262, i32 3
  %L281 = load <8 x i1>, ptr %Sl124
  %E282 = extractelement <4 x i1> zeroinitializer, i32 2
  br i1 %E282, label %CF826, label %CF848

CF848:                                            ; preds = %CF866
  %Cmp288 = fcmp olt <4 x double> undef, %I276
  %FC294 = fptosi double undef to i16
  %Cmp296 = icmp ule i16 %FC294, %B245
  store i16 %L218, ptr undef
  store i8 %L23, ptr %0
  %E320 = extractelement <4 x i1> %Sl223, i32 1
  %Cmp345 = icmp uge <1 x i32> undef, %Sl148
  store i16 %L196, ptr %1
  br label %CF893

CF893:                                            ; preds = %CF893, %CF848
  %Cmp361 = fcmp uge float undef, undef
  br i1 %Cmp361, label %CF893, label %CF906

CF906:                                            ; preds = %CF893
  store i16 -11933, ptr undef
  %Shuff379 = shufflevector <1 x i1> undef, <1 x i1> %Cmp345, <1 x i32> <i32 1>
  br label %CF850

CF850:                                            ; preds = %CF850, %CF906
  br i1 undef, label %CF850, label %CF925

CF925:                                            ; preds = %CF850
  store i16 %E266, ptr %1
  %Cmp413 = icmp ugt i8 %L257, undef
  store i16 %L30, ptr %1
  %Sl420 = select i1 %Sl171, <8 x i1> undef, <8 x i1> %L281
  store i16 %L90, ptr undef
  %FC469 = uitofp i1 %Cmp296 to float
  store i1 %Cmp413, ptr %Sl124
  br label %CF833

CF833:                                            ; preds = %CF833, %CF925
  store i8 %B207, ptr %0
  %E509 = extractelement <8 x i1> %L211, i32 7
  br i1 %E509, label %CF833, label %CF882

CF882:                                            ; preds = %CF833
  store i1 %Sl271, ptr %Sl124
  br label %CF852

CF852:                                            ; preds = %CF896, %CF882
  store i1 %Sl74, ptr %Sl124
  br label %CF896

CF896:                                            ; preds = %CF852
  %E576 = extractelement <4 x i1> %Cmp288, i32 3
  br i1 %E576, label %CF852, label %CF890

CF890:                                            ; preds = %CF896
  %Sl581 = select i1 undef, float undef, float %FC469
  unreachable
}
