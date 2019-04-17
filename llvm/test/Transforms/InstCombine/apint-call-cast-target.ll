; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"

define i32 @main() {
; CHECK-LABEL: @main(
; CHECK: %[[call:.*]] = call i7* @ctime(i999* null)
; CHECK: %[[cast:.*]] = ptrtoint i7* %[[call]] to i32
; CHECK: ret i32 %[[cast]]
entry:
	%tmp = call i32 bitcast (i7* (i999*)* @ctime to i32 (i99*)*)( i99* null )
	ret i32 %tmp
}

define i7* @ctime(i999*) {
; CHECK-LABEL: define i7* @ctime(
; CHECK: %[[call:.*]] = call i32 @main()
; CHECK: %[[cast:.*]] = inttoptr i32 %[[call]] to i7*
entry:
	%tmp = call i7* bitcast (i32 ()* @main to i7* ()*)( )
	ret i7* %tmp
}
