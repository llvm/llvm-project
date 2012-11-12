; RUN: opt < %s -instcombine -S | grep call | not grep bitcast

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"


define i32 @main() {
entry:
	%tmp = call i32 bitcast (i7* (i999*)* @ctime to i32 (i99*)*)( i99* null )
	ret i32 %tmp
}

define i7* @ctime(i999*) {
entry:
	%tmp = call i7* bitcast (i32 ()* @main to i7* ()*)( )
	ret i7* %tmp
}
