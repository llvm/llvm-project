; RUN: llc < %s -mtriple=x86_64-unknown-linux | FileCheck --check-prefix=NO-ADDRSIG %s
; RUN: llc < %s -mtriple=x86_64-unknown-linux -addrsig | FileCheck %s
; RUN: llc %s -filetype=obj -mtriple=x86_64-unknown-linux -addrsig -o %t
; RUN: llvm-readobj --addrsig %t | FileCheck %s --check-prefix=SYM

; NO-ADDRSIG-NOT: .addrsig

; SYM:      Addrsig [
; SYM-NEXT:   Sym: f1
; SYM-NEXT:   Sym: metadata_f2
; SYM-NEXT:   Sym: g1
; SYM-NEXT:   Sym: a1
; SYM-NEXT:   Sym: i1
; SYM-NEXT: ]

; CHECK: .addrsig

; CHECK: .addrsig_sym f1
define void()* @f1() {
  %f1 = bitcast void()* ()* @f1 to i8*
  %f2 = bitcast void()* ()* @f2 to i8*
  %f3 = bitcast void()* @f3 to i8*
  %g1 = bitcast i32* @g1 to i8*
  %g2 = bitcast i32* @g2 to i8*
  %g3 = bitcast i32* @g3 to i8*
  %dllimport = bitcast i32* @dllimport to i8*
  %tls = bitcast i32* @tls to i8*
  %a1 = bitcast i32* @a1 to i8*
  %a2 = bitcast i32* @a2 to i8*
  %i1 = bitcast void()* @i1 to i8*
  %i2 = bitcast void()* @i2 to i8*
  call void @llvm.dbg.value(metadata i8* bitcast (void()* @metadata_f1 to i8*), metadata !5, metadata !DIExpression()), !dbg !7
  call void @llvm.dbg.value(metadata i8* bitcast (void()* @metadata_f2 to i8*), metadata !5, metadata !DIExpression()), !dbg !7
  call void @f4(i8* bitcast (void()* @metadata_f2 to i8*))
  unreachable
}

declare void @f4(i8*) unnamed_addr

;; f1 is unreferenced, so this directive does not emit an entry.
; CHECK: .addrsig_sym metadata_f1
declare void @metadata_f1()

; CHECK: .addrsig_sym metadata_f2
declare void @metadata_f2()

; CHECK-NOT: .addrsig_sym f2
define internal void()* @f2() local_unnamed_addr {
  unreachable
}

; CHECK-NOT: .addrsig_sym f3
declare void @f3() unnamed_addr

; CHECK: .addrsig_sym g1
@g1 = global i32 0
; CHECK-NOT: .addrsig_sym g2
@g2 = internal local_unnamed_addr global i32 0
; CHECK-NOT: .addrsig_sym g3
@g3 = external unnamed_addr global i32

; CHECK-NOT: .addrsig_sym unref
@unref = external global i32

; CHECK-NOT: .addrsig_sym dllimport
@dllimport = external dllimport global i32

; CHECK-NOT: .addrsig_sym tls
@tls = thread_local global i32 0

; CHECK: .addrsig_sym a1
@a1 = alias i32, i32* @g1
; CHECK-NOT: .addrsig_sym a2
@a2 = internal local_unnamed_addr alias i32, i32* @g2

; CHECK: .addrsig_sym i1
@i1 = ifunc void(), void()* ()* @f1
; CHECK-NOT: .addrsig_sym i2
@i2 = internal local_unnamed_addr ifunc void(), void()* ()* @f2

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "a", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !0)
!4 = !DILocation(line: 0, scope: !3)
!5 = !DILocalVariable(scope: !6)
!6 = distinct !DISubprogram(scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !0)
!7 = !DILocation(line: 0, scope: !6, inlinedAt: !4)
