; RUN: llc -mtriple=xtensa --filetype=obj %s -o - | llvm-readelf --unwind - | FileCheck %s --check-prefix=XTENSA
; RUN: llc -mtriple=xtensa -mattr=+windowed --filetype=obj %s -o - | llvm-readelf --unwind - | FileCheck %s --check-prefix=XTENSA-WINDOWED

define dso_local i32 @test(i32 noundef %a) {
entry:
  ret i32 %a
}

; XTENSA: return_address_register: 0
; XTENSA-WINDOWED: return_address_register: 0
