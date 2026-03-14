; RUN: llc < %s
; rdar://7759395

%0 = type { i32, i32 }

define void @t(ptr, i32, i32, i32, i32) nounwind {
  tail call void @llvm.dbg.value(metadata ptr %0, i64 0, metadata !0, metadata !DIExpression()), !dbg !DILocation(scope: !1)
  unreachable
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

; !0 should conform to the format of DIVariable.
!0 = !DILocalVariable(name: "a", arg: 1, scope: !1)
!1 = distinct !DISubprogram()
