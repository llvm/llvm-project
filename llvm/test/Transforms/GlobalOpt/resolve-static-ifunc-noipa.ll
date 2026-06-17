; RUN: opt --passes=globalopt -o - -S < %s | FileCheck %s

; `noipa` resolvers are not eligible for inspection

@static.ifunc = internal ifunc void (), ptr @static.resolver

define ptr @static.resolver() noipa {
  ret ptr @static._Msimd
}
define void @static._Msimd() {
  ret void
}
define void @static.default() {
  ret void
}

define void @caller() {
  ; CHECK: call void @static.ifunc()
  call void @static.ifunc()
  ret void
}
