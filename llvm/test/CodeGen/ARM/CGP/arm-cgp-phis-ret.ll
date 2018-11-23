; RUN: llc -mtriple=thumbv7m -arm-disable-cgp=false %s -o - | FileCheck %s --check-prefix=CHECK-COMMON
; RUN: llc -mtriple=thumbv8m.main -arm-disable-cgp=false %s -o - | FileCheck %s --check-prefix=CHECK-COMMON
; RUN: llc -mtriple=thumbv8m.main -arm-disable-cgp=false -arm-enable-scalar-dsp=true -mcpu=cortex-m33 %s -o - | FileCheck %s --check-prefix=CHECK-COMMON
; RUN: llc -mtriple=thumbv7em %s -arm-disable-cgp=false -arm-enable-scalar-dsp=true -arm-enable-scalar-dsp-imms=true -o - | FileCheck %s --check-prefix=CHECK-COMMON

; Test that ARMCodeGenPrepare can handle:
; - loops
; - call operands
; - call return values
; - ret instructions
; We use nuw on the arithmetic instructions to avoid complications.

; Check that the arguments are extended but then nothing else is.
; This also ensures that the pass can handle loops.
; CHECK-COMMON-LABEL: phi_feeding_phi_args
; CHECK-COMMON: uxtb
; CHECK-COMMON: uxtb
; CHECK-NOT: uxtb
define void @phi_feeding_phi_args(i8 %a, i8 %b) {
entry:
  %0 = icmp ugt i8 %a, %b
  br i1 %0, label %preheader, label %empty

empty:
  br label %preheader

preheader:
  %1 = phi i8 [ %a, %entry ], [ %b, %empty ]
  br label %loop

loop:
  %val = phi i8 [ %1, %preheader ], [ %inc2, %if.end ]
  %cmp = icmp ult i8 %val, 254
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %inc = sub nuw i8 %val, 2
  br label %if.end

if.else:
  %inc1 = shl nuw i8 %val, 1
  br label %if.end

if.end:
  %inc2 = phi i8 [ %inc, %if.then], [ %inc1, %if.else ]
  %cmp1 = icmp eq i8 %inc2, 255
  br i1 %cmp1, label %exit, label %loop

exit:
  ret void
}

; Same as above, but as the args are zeroext, we shouldn't see any uxts.
; CHECK-COMMON-LABEL: phi_feeding_phi_zeroext_args
; CHECK-COMMON-NOT: uxt
define void @phi_feeding_phi_zeroext_args(i8 zeroext %a, i8 zeroext %b) {
entry:
  %0 = icmp ugt i8 %a, %b
  br i1 %0, label %preheader, label %empty

empty:
  br label %preheader

preheader:
  %1 = phi i8 [ %a, %entry ], [ %b, %empty ]
  br label %loop

loop:
  %val = phi i8 [ %1, %preheader ], [ %inc2, %if.end ]
  %cmp = icmp ult i8 %val, 254
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %inc = sub nuw i8 %val, 2
  br label %if.end

if.else:
  %inc1 = shl nuw i8 %val, 1
  br label %if.end

if.end:
  %inc2 = phi i8 [ %inc, %if.then], [ %inc1, %if.else ]
  %cmp1 = icmp eq i8 %inc2, 255
  br i1 %cmp1, label %exit, label %loop

exit:
  ret void
}

; Just check that phis also work with i16s.
; CHECK-COMMON-LABEL: phi_i16:
; CHECK-COMMON-NOT:   uxt
define void @phi_i16() {
entry:
  br label %loop

loop:
  %val = phi i16 [ 0, %entry ], [ %inc2, %if.end ]
  %cmp = icmp ult i16 %val, 128
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %inc = add nuw i16 %val, 2
  br label %if.end

if.else:
  %inc1 = add nuw i16 %val, 1
  br label %if.end

if.end:
  %inc2 = phi i16 [ %inc, %if.then], [ %inc1, %if.else ]
  %cmp1 = icmp ult i16 %inc2, 253
  br i1 %cmp1, label %loop, label %exit

exit:
  ret void
}

; CHECK-COMMON-LABEL: ret_i8
; CHECK-COMMON-NOT:   uxt
define i8 @ret_i8() {
entry:
  br label %loop

loop:
  %val = phi i8 [ 0, %entry ], [ %inc2, %if.end ]
  %cmp = icmp ult i8 %val, 128
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %inc = add nuw i8 %val, 2
  br label %if.end

if.else:
  %inc1 = add nuw i8 %val, 1
  br label %if.end

if.end:
  %inc2 = phi i8 [ %inc, %if.then], [ %inc1, %if.else ]
  %cmp1 = icmp ult i8 %inc2, 253
  br i1 %cmp1, label %exit, label %loop

exit:
  ret i8 %inc2
}

; CHECK-COMMON-LABEL: phi_multiple_undefs
; CHECK-COMMON-NOT:   uxt
define i16 @phi_multiple_undefs(i16 zeroext %arg) {
entry:
  br label %loop

loop:
  %val = phi i16 [ undef, %entry ], [ %inc2, %if.end ]
  %cmp = icmp ult i16 %val, 128
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %inc = add nuw i16 %val, 2
  br label %if.end

if.else:
  %inc1 = add nuw i16 %val, 1
  br label %if.end

if.end:
  %inc2 = phi i16 [ %inc, %if.then], [ %inc1, %if.else ]
  %unrelated = phi i16 [ undef, %if.then ], [ %arg, %if.else ]
  %cmp1 = icmp ult i16 %inc2, 253
  br i1 %cmp1, label %loop, label %exit

exit:
  ret i16 %unrelated
}

; CHECK-COMMON-LABEL: promote_arg_return
; CHECK-COMMON-NOT: uxt
; CHECK-COMMON: strb
define i16 @promote_arg_return(i16 zeroext %arg1, i16 zeroext %arg2, i8* %res) {
  %add = add nuw i16 %arg1, 15
  %mul = mul nuw nsw i16 %add, 3
  %cmp = icmp ult i16 %mul, %arg2
  %conv = zext i1 %cmp to i8
  store i8 %conv, i8* %res
  ret i16 %arg1
}
