; REQUIRES: x86-registered-target

; RUN: rm -rf %t.dir && split-file %s %t.dir
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %t.dir/lib.ll -o %t.dir/lib.bc
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %t.dir/main.ll -o %t.dir/main.bc
; RUN: llvm-lto2 run -save-temps %t.dir/lib.bc %t.dir/main.bc -o %t.dir/summary \
; RUN:   -r=%t.dir/lib.bc,_Z11public_funcv,plx \
; RUN:   -r=%t.dir/lib.bc,syscall, \
; RUN:   -r=%t.dir/lib.bc,_ZN12_GLOBAL__N_113internal_funcEv.35df87b54cddf81e734207bfc5eea57a,pl \
; RUN:   -r=%t.dir/main.bc,main,plx \
; RUN:   -r=%t.dir/main.bc,_Z11public_funcv,l
; RUN: llvm-dis %t.dir/summary.2.4.opt.bc -o - | FileCheck %s

; RUN: llvm-lto2 run -thinlto-distributed-indexes %t.dir/lib.bc %t.dir/main.bc \
; RUN:   -o %t.dir/distidx \
; RUN:   -r=%t.dir/lib.bc,_Z11public_funcv,plx \
; RUN:   -r=%t.dir/lib.bc,syscall, \
; RUN:   -r=%t.dir/lib.bc,_ZN12_GLOBAL__N_113internal_funcEv.35df87b54cddf81e734207bfc5eea57a,pl \
; RUN:   -r=%t.dir/main.bc,main,plx \
; RUN:   -r=%t.dir/main.bc,_Z11public_funcv,l
; RUN: llvm-bcanalyzer -dump %t.dir/lib.bc.thinlto.bc | FileCheck %s --check-prefix=DIST

; Verify that the type test is NOT lowered to an unconditional trap,
; and that the initialization path (which calls syscall) is preserved.
; CHECK: define hidden noundef i32 @main()
; CHECK:      %[[LOAD:.*]] = load i1, ptr @_ZN12_GLOBAL__N_18lazy_valE.1.llvm.{{.*}}
; CHECK-NEXT: br i1 %[[LOAD]], label %{{.*}}, label %[[LABEL:.*]]
; CHECK:      [[LABEL]]:
; CHECK:      call i64 (i64, ...) @syscall(i64 noundef 186)
; CHECK-NOT:  call void @llvm.ubsantrap

; Verify that the distributed index records CFI_FUNCTION_DEFS with the
; 3-field format [GUID, strtab_offset, name_length]. The promoted internal
; function is listed with its stable pre-promotion GUID.
; DIST:     <CFI_FUNCTION_DEFS op0={{[-0-9]+}} op1=0 op2=67/>
; DIST: <STRTAB_BLOCK
; DIST-NEXT:   <BLOB abbrevid=4/> blob data = '_ZN12_GLOBAL__N_113internal_funcEv.35df87b54cddf81e734207bfc5eea57a_ZTSFjvE'

;--- lib.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@_ZN12_GLOBAL__N_18lazy_valE.1 = internal unnamed_addr global i1 false, align 4

define hidden void @_Z11public_funcv() local_unnamed_addr #0 !type !10 !type !11 {
  %1 = load i1, ptr @_ZN12_GLOBAL__N_18lazy_valE.1, align 1
  %2 = zext i1 %1 to i8
  %3 = trunc nuw i8 %2 to i1
  br i1 %3, label %9, label %4

4:                                                ; preds = %0
  %5 = tail call i1 @llvm.type.test(ptr nonnull @_ZN12_GLOBAL__N_113internal_funcEv, metadata !"_ZTSFjvE")
  br i1 %5, label %7, label %6
6:                                                ; preds = %4
  tail call void @llvm.ubsantrap(i8 2) #5
  unreachable

7:                                                ; preds = %4
  %8 = tail call i64 (i64, ...) @syscall(i64 noundef 186)
  store i1 true, ptr @_ZN12_GLOBAL__N_18lazy_valE.1, align 1
  br label %9

9:                                                ; preds = %0, %7
  ret void
}

define internal noundef i32 @_ZN12_GLOBAL__N_113internal_funcEv() #1 !type !15 !type !16 {
  %1 = tail call i64 (i64, ...) @syscall(i64 noundef 186)
  %2 = trunc i64 %1 to i32
  ret i32 %2
}

declare i1 @llvm.type.test(ptr, metadata) #2
declare void @llvm.ubsantrap(i8 immarg) #3
declare i64 @syscall(i64 noundef, ...) #4

attributes #0 = { alwaysinline mustprogress nounwind uwtable "target-cpu"="x86-64" }
attributes #1 = { mustprogress nounwind uwtable "target-cpu"="x86-64" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #4 = { nounwind "target-cpu"="x86-64" }
attributes #5 = { nomerge noreturn nounwind }

!10 = !{i64 0, !"_ZTSFvvE"}
!11 = !{i64 0, !"_ZTSFvvE.generalized"}
!15 = !{i64 0, !"_ZTSFjvE"}
!16 = !{i64 0, !"_ZTSFjvE.generalized"}

;--- main.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define hidden noundef i32 @main() local_unnamed_addr #0 !type !8 !type !9 {
  tail call void @_Z11public_funcv()
  ret i32 0
}

declare !type !11 !type !12 dso_local void @_Z11public_funcv() local_unnamed_addr #1

attributes #0 = { mustprogress norecurse uwtable "target-cpu"="x86-64" }
attributes #1 = { "target-cpu"="x86-64" }

!8 = !{i64 0, !"_ZTSFivE"}
!9 = !{i64 0, !"_ZTSFivE.generalized"}
!11 = !{i64 0, !"_ZTSFvvE"}
!12 = !{i64 0, !"_ZTSFvvE.generalized"}

