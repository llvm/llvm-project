; RUN: llc < %s -mtriple=i386-apple-darwin10 | FileCheck %s
; RUN: llc < %s -mtriple=i386-apple-darwin10 -regalloc=basic | FileCheck %s
; rdar://6509240

; CHECK: os_clock
; CHECK-NOT: movaps

	%0 = type { %struct.TValue }		; type %0
	%1 = type { %struct.L_Umaxalign, i32, ptr }		; type %1
	%struct.CallInfo = type { ptr, ptr, ptr, ptr, i32, i32 }
	%struct.GCObject = type { %struct.lua_State }
	%struct.L_Umaxalign = type { double }
	%struct.Mbuffer = type { ptr, i32, i32 }
	%struct.Node = type { %struct.TValue, %struct.TKey }
	%struct.TKey = type { %1 }
	%struct.TString = type { %struct.anon }
	%struct.TValue = type { %struct.L_Umaxalign, i32 }
	%struct.Table = type { ptr, i8, i8, i8, i8, ptr, ptr, ptr, ptr, ptr, i32 }
	%struct.UpVal = type { ptr, i8, i8, ptr, %0 }
	%struct.anon = type { ptr, i8, i8, i8, i32, i32 }
	%struct.global_State = type { %struct.stringtable, ptr, ptr, i8, i8, i32, ptr, ptr, ptr, ptr, ptr, ptr, %struct.Mbuffer, i32, i32, i32, i32, i32, i32, ptr, %struct.TValue, ptr, %struct.UpVal, [9 x ptr], [17 x ptr] }
	%struct.lua_Debug = type { i32, ptr, ptr, ptr, ptr, i32, i32, i32, i32, [60 x i8], i32 }
	%struct.lua_State = type { ptr, i8, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i16, i16, i8, i8, i32, i32, ptr, %struct.TValue, %struct.TValue, ptr, ptr, ptr, i32 }
	%struct.lua_longjmp = type { ptr, [18 x i32], i32 }
	%struct.stringtable = type { ptr, i32, i32 }
@llvm.used = appending global [1 x ptr] [ptr @os_clock], section "llvm.metadata"		; <ptr> [#uses=0]

define i32 @os_clock(ptr nocapture %L) nounwind ssp {
entry:
	%0 = tail call i32 @"\01_clock$UNIX2003"() nounwind		; <i32> [#uses=1]
	%1 = uitofp i32 %0 to double		; <double> [#uses=1]
	%2 = fdiv double %1, 1.000000e+06		; <double> [#uses=1]
	%3 = getelementptr %struct.lua_State, ptr %L, i32 0, i32 4		; <ptr> [#uses=3]
	%4 = load ptr, ptr %3, align 4		; <ptr> [#uses=2]
	%5 = getelementptr %struct.TValue, ptr %4, i32 0, i32 0, i32 0		; <ptr> [#uses=1]
	store double %2, ptr %5, align 4
	%6 = getelementptr %struct.TValue, ptr %4, i32 0, i32 1		; <ptr> [#uses=1]
	store i32 3, ptr %6, align 4
	%7 = load ptr, ptr %3, align 4		; <ptr> [#uses=1]
	%8 = getelementptr %struct.TValue, ptr %7, i32 1		; <ptr> [#uses=1]
	store ptr %8, ptr %3, align 4
	ret i32 1
}

declare i32 @"\01_clock$UNIX2003"()
