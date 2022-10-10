; RUN: opt -S -mtriple=aarch64-linux-gnu -aarch64-sme-abi %s | FileCheck %s
; RUN: opt -S -mtriple=aarch64-linux-gnu -aarch64-sme-abi -aarch64-sme-abi %s | FileCheck %s

declare void @shared_za_callee() "aarch64_pstate_za_shared"

define void @private_za() "aarch64_pstate_za_new" {
; CHECK-LABEL: @private_za(
; CHECK-NEXT:  prelude:
; CHECK-NEXT:    [[TPIDR2:%.*]] = call i64 @llvm.aarch64.sme.get.tpidr2()
; CHECK-NEXT:    [[CMP:%.*]] = icmp ne i64 [[TPIDR2]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[SAVE_ZA:%.*]], label [[TMP0:%.*]]
; CHECK:       save.za:
; CHECK-NEXT:    call void @__arm_tpidr2_save()
; CHECK-NEXT:    call void @llvm.aarch64.sme.set.tpidr2(i64 0)
; CHECK-NEXT:    br label [[TMP0]]
; CHECK:       0:
; CHECK-NEXT:    call void @llvm.aarch64.sme.za.enable()
; CHECK-NEXT:    call void @shared_za_callee()
; CHECK-NEXT:    call void @llvm.aarch64.sme.za.disable()
; CHECK-NEXT:    ret void
;
  call void @shared_za_callee()
  ret void
}

define i32 @private_za_multiple_exit(i32 %a, i32 %b, i64 %cond) "aarch64_pstate_za_new" {
; CHECK-LABEL: @private_za_multiple_exit(
; CHECK-NEXT:  prelude:
; CHECK-NEXT:    [[TPIDR2:%.*]] = call i64 @llvm.aarch64.sme.get.tpidr2()
; CHECK-NEXT:    [[CMP:%.*]] = icmp ne i64 [[TPIDR2]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[SAVE_ZA:%.*]], label [[ENTRY:%.*]]
; CHECK:       save.za:
; CHECK-NEXT:    call void @__arm_tpidr2_save()
; CHECK-NEXT:    call void @llvm.aarch64.sme.set.tpidr2(i64 0)
; CHECK-NEXT:    br label [[ENTRY]]
; CHECK:       entry:
; CHECK-NEXT:    call void @llvm.aarch64.sme.za.enable()
; CHECK-NEXT:    [[TOBOOL:%.*]] = icmp eq i64 [[COND:%.*]], 1
; CHECK-NEXT:    br i1 [[TOBOOL]], label [[IF_ELSE:%.*]], label [[IF_END:%.*]]
; CHECK:       if.else:
; CHECK-NEXT:    [[ADD:%.*]] = add i32 [[A:%.*]], [[B:%.*]]
; CHECK-NEXT:    call void @llvm.aarch64.sme.za.disable()
; CHECK-NEXT:    ret i32 [[ADD]]
; CHECK:       if.end:
; CHECK-NEXT:    [[SUB:%.*]] = sub i32 [[A]], [[B]]
; CHECK-NEXT:    call void @llvm.aarch64.sme.za.disable()
; CHECK-NEXT:    ret i32 [[SUB]]
;
entry:
  %tobool = icmp eq i64 %cond, 1
  br i1 %tobool, label %if.else, label %if.end

if.else:
  %add = add i32 %a, %b
  ret i32 %add

if.end:
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; CHECK: declare "aarch64_pstate_sm_compatible" void @__arm_tpidr2_save()
