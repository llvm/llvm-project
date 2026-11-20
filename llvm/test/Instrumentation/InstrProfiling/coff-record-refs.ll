; RUN: split-file %s %t
; RUN: opt < %t/main.ll -mtriple=x86_64-pc-windows-msvc -passes=instrprof -S | FileCheck %s -DALIGN=8 --implicit-check-not='.lprfp$M'
; RUN: opt < %t/main.ll -mtriple=i686-pc-windows-msvc -passes=instrprof -S | FileCheck %s -DALIGN=4 --implicit-check-not='.lprfp$M'
; RUN: cat %t/main.ll %t/vp.ll | opt -mtriple=x86_64-pc-windows-msvc -passes=instrprof -S | FileCheck %s --check-prefix=WEAK
; RUN: opt < %t/main.ll -mtriple=x86_64-pc-windows-gnu -passes=instrprof -S | FileCheck %s --check-prefix=NO-REF
; RUN: opt < %t/main.ll -mtriple=x86_64-unknown-linux -passes=instrprof -S | FileCheck %s --check-prefix=NO-REF
; RUN: opt < %t/main.ll -mtriple=x86_64-pc-windows-msvc -passes=instrprof -profile-correlate=binary -S | FileCheck %s --check-prefix=NO-REF
; RUN: opt < %t/main.ll -mtriple=x86_64-pc-windows-msvc -passes=instrprof -profile-correlate=debug-info -S | FileCheck %s --check-prefix=NO-REF
; RUN: opt < %t/names.ll -mtriple=x86_64-pc-windows-msvc -passes=instrprof -S | FileCheck %s --check-prefix=NO-REF

; Two increments of one ordinary record still produce exactly one slot.
; CHECK: @__profd_plain = private global {{.*}}, section ".lprfd$0", align 8
; CHECK: @__profd_plain.ref = private constant ptr @__profd_plain, section ".lprfp$M", align [[ALIGN]]
; The slot and the record follow the same prevailing counter group.
; CHECK: @__profd_inline = internal global {{.*}}, section ".lprfd$0", comdat($__profc_inline), align 8
; CHECK: @__profd_inline.ref = private constant ptr @__profd_inline, section ".lprfp$M", comdat($__profc_inline), align [[ALIGN]]
; CHECK: @llvm.compiler.used = {{.*}}ptr @__profd_plain.ref, ptr @__profd_inline.ref]
; CHECK: @llvm.used = {{.*}}ptr @__llvm_profile_record_refs_v1]

; When code can reference a weak record its slot must coalesce with the record,
; even if the original function had no explicit COMDAT.
; WEAK: @__profd_plain = weak hidden global {{.*}}, section ".lprfd$0", comdat, align 8
; WEAK: @__profd_plain.ref = private constant ptr @__profd_plain, section ".lprfp$M", comdat($__profd_plain), align 8

; NO-REF-NOT: .lprfp$M
; NO-REF-NOT: .lprfd$0
; NO-REF-NOT: __llvm_profile_record_refs_v1

;--- main.ll
@__profn_plain = weak hidden constant [5 x i8] c"plain"
@__profn_inline = linkonce_odr hidden constant [6 x i8] c"inline"
$inline = comdat any

define weak void @plain() {
  call void @llvm.instrprof.increment(ptr @__profn_plain, i64 1234, i32 2, i32 0)
  call void @llvm.instrprof.increment(ptr @__profn_plain, i64 1234, i32 2, i32 1)
  ret void
}

define linkonce_odr void @inline() comdat {
  call void @llvm.instrprof.increment(ptr @__profn_inline, i64 5678, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)

;--- vp.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"EnableValueProfiling", i32 1}

;--- names.ll
@__profn_unused = private constant [6 x i8] c"unused"
@__llvm_coverage_names = internal constant [1 x ptr] [ptr @__profn_unused]
