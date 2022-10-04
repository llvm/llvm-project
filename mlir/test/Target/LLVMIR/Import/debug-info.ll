; RUN: mlir-translate -import-llvm -mlir-print-debuginfo -split-input-file %s | FileCheck %s

; CHECK: #[[$UNKNOWNLOC:.+]] = loc(unknown)
; CHECK-LABEL: @unknown(
define i32 @unknown(i32 %0) {
entry:
  br label %next
end:
  ; CHECK: ^{{.*}}(%{{.+}}: i32 loc(unknown)):
  %1 = phi i32 [ %2, %next ]
  ret i32 %1
next:
  ; CHECK: = llvm.mul %{{.+}}, %{{.+}} : i32 loc(#[[$UNKNOWNLOC:.+]])
  %2 = mul i32 %0, %0
  br label %end
}

; // -----

; CHECK-LABEL: @known_loc(
define i32 @known_loc(i32 %0) {
entry:
  br label %next
end:
  ; CHECK: ^{{.*}}(%{{.+}}: i32 loc("known_loc.cpp":5:2)):
  %1 = phi i32 [ %2, %next ], !dbg !4
  ret i32 %1
next:
  ; CHECK: = llvm.mul %{{.+}}, %{{.+}} : i32 loc(#[[LOC:.+]])
  %2 = mul i32 %0, %0, !dbg !5
  br label %end
}
; CHECK: #[[LOC]] = loc("known_loc.cpp":8:3)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !2)
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !DIFile(filename: "known_loc.cpp", directory: "/")
!3 = distinct !DISubprogram(name: "known_loc", scope: !0, file: !2, line: 1, scopeLine: 1, unit: !0)
!4 = !DILocation(line: 5, column: 2, scope: !3)
!5 = !DILocation(line: 8, column: 3, scope: !3)
