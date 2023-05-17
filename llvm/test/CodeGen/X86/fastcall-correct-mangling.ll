; RUN: llc < %s -mtriple=i386-unknown-mingw32 | \
; RUN: FileCheck --check-prefix=CHECK32 %s

; RUN: llc < %s -mtriple=i386-unknown-win32 | \
; RUN: FileCheck --check-prefix=CHECK32 %s

; RUN: llc < %s -mtriple=x86_64-unknown-mingw32 | \
; RUN: FileCheck --check-prefix=CHECK64 %s

; RUN: llc < %s -mtriple=x86_64-unknown-mingw32 | \
; RUN: FileCheck --check-prefix=CHECK64 %s

; Check that a fastcall function gets correct mangling

define x86_fastcallcc void @func(i64 %X, i8 %Y, i8 %G, i16 %Z) {
; CHECK32-LABEL: {{^}}@func@20:
; CHECK64-LABEL: {{^}}func:
        ret void
}

define x86_fastcallcc i32 @"\01DoNotMangle"(i32 %a) {
; CHECK32-LABEL: {{^}}DoNotMangle:
; CHECK64-LABEL: {{^}}DoNotMangle:
entry:
  ret i32 %a
}

define private x86_fastcallcc void @dontCrash() {
; The name is fairly arbitrary since it is private. Just don't crash.
; CHECK32-LABEL: {{^}}L@dontCrash@0:
; CHECK64-LABEL: {{^}}.LdontCrash:
  ret void
}

@alias = alias void(i64, i8, i8, i16), ptr @func
; CHECK32-LABEL: {{^}}.set @alias@20, @func@20
; CHECK64-LABEL: {{^}}.set alias, func
