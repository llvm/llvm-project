; RUN: llc -stop-after=systemz-isel -mtriple=s390x-ibm-linux < %s -o - | FileCheck -check-prefix=CHECK-DAGCOMBINE %s
; RUN: llc -stop-after=finalize-isel -mtriple=s390x-ibm-linux < %s -o - | FileCheck -check-prefix=CHECK-CUSTOMINSERT %s
; CHECK-DAGCOMBINE:   bb.0.entry:
; CHECK-DAGCOMBINE:     MOVE_SG_DAG %stack.0.StackGuardSlot, 0
; CHECK-DAGCOMBINE:     COMPARE_SG_BRIDGE %stack.0.StackGuardSlot, 0, implicit-def $cc
; CHECK-CUSTOMINSERT: bb.0.entry
; CHECK-CUSTOMINSERT:   early-clobber %6:addr64bit = MOVE_SG %stack.0.StackGuardSlot, 0
; CHECK_CUSTOMINSERT: bb.3.entry
; CHECK-CUSTOMINSERT: early-clobber %10:addr64bit = COMPARE_SG %stack.0.StackGuardSlot, 0, implicit-def $cc

define dso_local signext i32 @stack_guard_pseudo_check(i32 %argc, ptr %argv) #0 {
entry:
  %Buffer = alloca [8 x i8], align 1
  call void @llvm.memset.p0.i64(ptr align 1 %Buffer, i8 0, i64 8, i1 false)
  %arraydecay = getelementptr inbounds [8 x i8], ptr %Buffer, i64 0, i64 0
  %call = call ptr @strcpy(ptr noundef %arraydecay, ptr noundef %argv)
  ret i32 0
}

declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare ptr @strcpy(ptr noundef, ptr noundef)

attributes #0 = { ssp }
