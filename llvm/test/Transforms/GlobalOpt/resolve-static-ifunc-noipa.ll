; RUN: opt --passes=globalopt -o - -S < %s | FileCheck %s

; `noipa` resolvers are not eligible for inspection

@static.ifunc = internal ifunc void (), ptr @static.resolver

define ptr @static.resolver() {
  ret ptr @static._Msimd
}
define void @static._Msimd() {
  ret void
}
define void @static.default() {
  ret void
}

@static_noipa.ifunc = internal ifunc void (), ptr @static_noipa.resolver

define ptr @static_noipa.resolver() noipa {
  ret ptr @static._Msimd
}

define void @caller() {
  ; CHECK: call void @static._Msimd()
  call void @static.ifunc()
  ; CHECK: call void @static_noipa.ifunc()
  call void @static_noipa.ifunc()
  ret void
}
