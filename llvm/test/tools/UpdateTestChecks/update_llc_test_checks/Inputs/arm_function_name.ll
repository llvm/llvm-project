; Check that we accept functions with '$' in the name.
;
; RUN: llc -mtriple=arm64-unknown-linux < %s | FileCheck --prefi=LINUX %s
; RUN: llc -mtriple=armv7-apple-darwin < %s | FileCheck --prefix=DARWIN %s
; RUN: llc -mtriple=armv7-apple-ios < %s | FileCheck --prefix=IOS %s
;
define hidden i32 @"_Z54bar$ompvariant$bar"() {
entry:
  ret i32 2
}
