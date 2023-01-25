; RUN: opt -S -passes=objc-arc < %s | FileCheck %s

declare ptr @llvm.objc.loadWeak(ptr)
declare ptr @llvm.objc.loadWeakRetained(ptr)
declare ptr @llvm.objc.storeWeak(ptr, ptr)
declare ptr @llvm.objc.initWeak(ptr, ptr)
declare void @use_pointer(ptr)
declare void @callee()

; Basic redundant @llvm.objc.loadWeak elimination.

; CHECK:      define void @test0(ptr %p) {
; CHECK-NEXT:   %y = call ptr @llvm.objc.loadWeak(ptr %p)
; CHECK-NEXT:   call void @use_pointer(ptr %y)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test0(ptr %p) {
  %x = call ptr @llvm.objc.loadWeak(ptr %p)
  %y = call ptr @llvm.objc.loadWeak(ptr %p)
  call void @use_pointer(ptr %y)
  ret void
}

; DCE the @llvm.objc.loadWeak.

; CHECK:      define void @test1(ptr %p) {
; CHECK-NEXT:   %y = call ptr @llvm.objc.loadWeakRetained(ptr %p)
; CHECK-NEXT:   call void @use_pointer(ptr %y)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test1(ptr %p) {
  %x = call ptr @llvm.objc.loadWeak(ptr %p)
  %y = call ptr @llvm.objc.loadWeakRetained(ptr %p)
  call void @use_pointer(ptr %y)
  ret void
}

; Basic redundant @llvm.objc.loadWeakRetained elimination.

; CHECK:      define void @test2(ptr %p) {
; CHECK-NEXT:   %x = call ptr @llvm.objc.loadWeak(ptr %p)
; CHECK-NEXT:   store i8 3, ptr %x
; CHECK-NEXT:   %1 = tail call ptr @llvm.objc.retain(ptr %x)
; CHECK-NEXT:   call void @use_pointer(ptr %x)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test2(ptr %p) {
  %x = call ptr @llvm.objc.loadWeak(ptr %p)
  store i8 3, ptr %x
  %y = call ptr @llvm.objc.loadWeakRetained(ptr %p)
  call void @use_pointer(ptr %y)
  ret void
}

; Basic redundant @llvm.objc.loadWeakRetained elimination, this time
; with a readonly call instead of a store.

; CHECK:      define void @test3(ptr %p) {
; CHECK-NEXT:   %x = call ptr @llvm.objc.loadWeak(ptr %p)
; CHECK-NEXT:   call void @use_pointer(ptr %x) [[RO:#[0-9]+]]
; CHECK-NEXT:   %1 = tail call ptr @llvm.objc.retain(ptr %x)
; CHECK-NEXT:   call void @use_pointer(ptr %x)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test3(ptr %p) {
  %x = call ptr @llvm.objc.loadWeak(ptr %p)
  call void @use_pointer(ptr %x) readonly
  %y = call ptr @llvm.objc.loadWeakRetained(ptr %p)
  call void @use_pointer(ptr %y)
  ret void
}

; A regular call blocks redundant weak load elimination.

; CHECK:      define void @test4(ptr %p) {
; CHECK-NEXT:   %x = call ptr @llvm.objc.loadWeak(ptr %p)
; CHECK-NEXT:   call void @use_pointer(ptr %x) [[RO]]
; CHECK-NEXT:   call void @callee()
; CHECK-NEXT:   %y = call ptr @llvm.objc.loadWeak(ptr %p)
; CHECK-NEXT:   call void @use_pointer(ptr %y)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test4(ptr %p) {
  %x = call ptr @llvm.objc.loadWeak(ptr %p)
  call void @use_pointer(ptr %x) readonly
  call void @callee()
  %y = call ptr @llvm.objc.loadWeak(ptr %p)
  call void @use_pointer(ptr %y)
  ret void
}

; Store to load forwarding.

; CHECK:      define void @test5(ptr %p, ptr %n) {
; CHECK-NEXT:   %1 = call ptr @llvm.objc.storeWeak(ptr %p, ptr %n)
; CHECK-NEXT:   call void @use_pointer(ptr %n)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test5(ptr %p, ptr %n) {
  call ptr @llvm.objc.storeWeak(ptr %p, ptr %n)
  %y = call ptr @llvm.objc.loadWeak(ptr %p)
  call void @use_pointer(ptr %y)
  ret void
}

; Store to load forwarding with objc_initWeak.

; CHECK:      define void @test6(ptr %p, ptr %n) {
; CHECK-NEXT:   %1 = call ptr @llvm.objc.initWeak(ptr %p, ptr %n)
; CHECK-NEXT:   call void @use_pointer(ptr %n)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test6(ptr %p, ptr %n) {
  call ptr @llvm.objc.initWeak(ptr %p, ptr %n)
  %y = call ptr @llvm.objc.loadWeak(ptr %p)
  call void @use_pointer(ptr %y)
  ret void
}

; Don't forward if there's a may-alias store in the way.

; CHECK:      define void @test7(ptr %p, ptr %n, ptr %q, ptr %m) {
; CHECK-NEXT:   call ptr @llvm.objc.initWeak(ptr %p, ptr %n)
; CHECK-NEXT:   call ptr @llvm.objc.storeWeak(ptr %q, ptr %m)
; CHECK-NEXT:   %y = call ptr @llvm.objc.loadWeak(ptr %p)
; CHECK-NEXT:   call void @use_pointer(ptr %y)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test7(ptr %p, ptr %n, ptr %q, ptr %m) {
  call ptr @llvm.objc.initWeak(ptr %p, ptr %n)
  call ptr @llvm.objc.storeWeak(ptr %q, ptr %m)
  %y = call ptr @llvm.objc.loadWeak(ptr %p)
  call void @use_pointer(ptr %y)
  ret void
}

; CHECK: attributes #0 = { nounwind }
; CHECK: attributes [[RO]] = { memory(read) }
