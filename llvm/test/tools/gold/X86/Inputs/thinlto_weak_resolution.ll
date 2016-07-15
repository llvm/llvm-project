target triple = "x86_64-unknown-linux-gnu"

; Alias are not optimized
@linkonceodralias = linkonce_odr alias void (), void ()* @linkonceodrfuncwithalias

; Alias are not optimized
@linkoncealias = linkonce alias void (), void ()* @linkoncefuncwithalias

; Function with an alias are not optimized
define linkonce_odr void @linkonceodrfuncwithalias() #0 {
entry:
  ret void
}

; Function with an alias are not optimized
define linkonce void @linkoncefuncwithalias() #0 {
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

