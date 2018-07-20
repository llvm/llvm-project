; RUN: llc < %s | FileCheck %s

; C++ source:
; void g();
; void f() {
;   try {
;     g();
;   } catch (...) {
;     g();
;   }
; }

; CHECK: "?f@@YAXXZ":                            # @"\01?f@@YAXXZ"
; CHECK:         .cv_fpo_proc    "?f@@YAXXZ" 0
; CHECK:         pushl   %ebp
; CHECK:         .cv_fpo_pushreg %ebp
; CHECK:         movl    %esp, %ebp
; CHECK:         .cv_fpo_setframe        %ebp
; ...
; CHECK:         .cv_fpo_endprologue
; CHECK:         retl

;       No FPO directives in the catchpad for now.
; CHECK: "?catch$2@?0??f@@YAXXZ@4HA":
; CHECK-NOT: .cv_fpo
; CHECK:         retl                            # CATCHRET
;   FIXME: This endproc is for the parent function. To get FPO data for
;   funclets we'd have to emit it first so the scopes don't nest.
; CHECK:         .cv_fpo_endproc

; CHECK-NOT: .cv_fpo_data
; CHECK: .cv_fpo_data "?f@@YAXXZ"
; CHECK-NOT: .cv_fpo_data

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.11.25508"

define void @"\01?f@@YAXXZ"() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) !dbg !8 {
entry:
  invoke void @"\01?g@@YAXXZ"()
          to label %try.cont unwind label %catch.dispatch, !dbg !11

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller, !dbg !13

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null, i32 64, i8* null], !dbg !13
  call void @"\01?g@@YAXXZ"() [ "funclet"(token %1) ], !dbg !14
  catchret from %1 to label %try.cont, !dbg !16

try.cont:                                         ; preds = %entry, %catch
  ret void, !dbg !17
}

declare void @"\01?g@@YAXXZ"() local_unnamed_addr #1

declare i32 @__CxxFrameHandler3(...)

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "1e688e4021b6626d049b9899f9d53a2a")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXXZ", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 4, column: 5, scope: !12)
!12 = distinct !DILexicalBlock(scope: !8, file: !1, line: 3, column: 7)
!13 = !DILocation(line: 5, column: 3, scope: !12)
!14 = !DILocation(line: 6, column: 5, scope: !15)
!15 = distinct !DILexicalBlock(scope: !8, file: !1, line: 5, column: 17)
!16 = !DILocation(line: 7, column: 3, scope: !15)
!17 = !DILocation(line: 8, column: 1, scope: !8)
