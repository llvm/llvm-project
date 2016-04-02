target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; Alias are not optimized
@linkoncealias = linkonce_odr alias void (), void ()* @linkonceodrfuncwithalias

; Function with an alias are not optimized
define linkonce_odr void @linkonceodrfuncwithalias() #0 {
entry:
  ret void
}

define linkonce_odr void @linkonceodrfunc() #0 {
entry:
  ret void
}
define linkonce void @linkoncefunc() #0 {
entry:
  ret void
}
define weak_odr void @weakodrfunc() #0 {
entry:
  ret void
}
define weak void @weakfunc() #0 {
entry:
  ret void
}

