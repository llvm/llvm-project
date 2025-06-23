; RUN: not opt -S -passes='metarenamer<invalid>' %s 2>&1 | FileCheck -check-prefix=ERR %s

; ERR: invalid metarenamer pass parameter 'invalid'

define i32 @0(i32, i32) {
  %3 = add i32 %0, %1
  ret i32 %3
}

