; RUN: split-file %s %t
; RUN: opt < %t/main.ll -mtriple=x86_64-pc-windows-msvc -passes=instrprof -S | FileCheck %s --implicit-check-not='.lprfr$M'
; RUN: opt < %t/main.ll -mtriple=aarch64-pc-windows-msvc -passes=instrprof -S | FileCheck %s --implicit-check-not='.lprfr$M'
; RUN: opt < %t/main.ll -mtriple=i686-pc-windows-msvc -passes=instrprof -S | FileCheck %s --check-prefix=NO-REF
; RUN: opt < %t/main.ll -mtriple=thumbv7-pc-windows-msvc -passes=instrprof -S | FileCheck %s --check-prefix=NO-REF
; RUN: opt < %t/main.ll -mtriple=arm64ec-pc-windows-msvc -passes=instrprof -S | FileCheck %s --check-prefix=NO-REF
; RUN: opt < %t/main.ll -mtriple=arm64x-pc-windows-msvc -passes=instrprof -S | FileCheck %s --check-prefix=NO-REF
; RUN: cat %t/main.ll %t/vp.ll | opt -mtriple=x86_64-pc-windows-msvc -passes=instrprof -S | FileCheck %s --check-prefix=WEAK
; RUN: opt < %t/main.ll -mtriple=x86_64-pc-windows-gnu -passes=instrprof -S | FileCheck %s --check-prefix=NO-REF
; RUN: opt < %t/main.ll -mtriple=x86_64-unknown-linux -passes=instrprof -S | FileCheck %s --check-prefix=NO-REF
; RUN: opt < %t/main.ll -mtriple=x86_64-pc-windows-msvc -passes=instrprof -profile-correlate=binary -S | FileCheck %s --check-prefix=NO-REF
; RUN: opt < %t/main.ll -mtriple=x86_64-pc-windows-msvc -passes=instrprof -profile-correlate=debug-info -S | FileCheck %s --check-prefix=NO-REF
; RUN: opt < %t/names.ll -mtriple=x86_64-pc-windows-msvc -passes=instrprof -S | FileCheck %s --check-prefix=NO-REF

; Two increments of one ordinary record still produce exactly one slot.
; CHECK: @__profd_plain = private global {{.*}}, section ".lprfd$0", align 8
; CHECK: @__profd_plain.rva = private constant i32 trunc (i64 sub (i64 ptrtoint (ptr @__profd_plain to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), section ".lprfr$M", align 4
; The slot and the record follow the same prevailing counter group.
; CHECK: @__profd_inline = internal global {{.*}}, section ".lprfd$0", comdat($__profc_inline), align 8
; CHECK: @__profd_inline.rva = private constant i32 {{.*}}ptr @__profd_inline {{.*}}, section ".lprfr$M", comdat($__profc_inline), align 4
; CHECK: @llvm.compiler.used = {{.*}}ptr @__profd_plain.rva, ptr @__profd_inline.rva]
; CHECK: @llvm.used = {{.*}}ptr @__llvm_profile_record_rvas_v2]

; When code can reference a weak record its slot must coalesce with the record,
; even if the original function had no explicit COMDAT.
; WEAK: @__profd_plain = weak hidden global {{.*}}, section ".lprfd$0", comdat, align 8
; WEAK: @__profd_plain.rva = private constant i32 {{.*}}, section ".lprfr$M", comdat($__profd_plain), align 4

; NO-REF-NOT: .lprfp$M
; NO-REF-NOT: .lprfr$M
; NO-REF-NOT: .lprfd$0
; NO-REF-NOT: __llvm_profile_record_

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
