; RUN: llc < %s -mtriple=i386-apple-darwin10 | FileCheck %s
; RUN: llc < %s -mtriple=i386-apple-darwin10 -regalloc=basic | FileCheck %s
; rdar://6509240

; CHECK: os_clock
; CHECK-NOT: movaps

	%0 = type { %struct.TValue }		; type %0
	%1 = type { %struct.L_Umaxalign, i32, %struct.Node* }		; type %1
	%struct.CallInfo = type { %struct.TValue*, %struct.TValue*, %struct.TValue*, i32*, i32, i32 }
	%struct.GCObject = type { %struct.lua_State }
	%struct.L_Umaxalign = type { double }
	%struct.Mbuffer = type { i8*, i32, i32 }
	%struct.Node = type { %struct.TValue, %struct.TKey }
	%struct.TKey = type { %1 }
	%struct.TString = type { %struct.anon }
	%struct.TValue = type { %struct.L_Umaxalign, i32 }
	%struct.Table = type { %struct.GCObject*, i8, i8, i8, i8, %struct.Table*, %struct.TValue*, %struct.Node*, %struct.Node*, %struct.GCObject*, i32 }
	%struct.UpVal = type { %struct.GCObject*, i8, i8, %struct.TValue*, %0 }
	%struct.anon = type { %struct.GCObject*, i8, i8, i8, i32, i32 }
	%struct.global_State = type { %struct.stringtable, i8* (i8*, i8*, i32, i32)*, i8*, i8, i8, i32, %struct.GCObject*, %struct.GCObject**, %struct.GCObject*, %struct.GCObject*, %struct.GCObject*, %struct.GCObject*, %struct.Mbuffer, i32, i32, i32, i32, i32, i32, i32 (%struct.lua_State*)*, %struct.TValue, %struct.lua_State*, %struct.UpVal, [9 x %struct.Table*], [17 x %struct.TString*] }
	%struct.lua_Debug = type { i32, i8*, i8*, i8*, i8*, i32, i32, i32, i32, [60 x i8], i32 }
	%struct.lua_State = type { %struct.GCObject*, i8, i8, i8, %struct.TValue*, %struct.TValue*, %struct.global_State*, %struct.CallInfo*, i32*, %struct.TValue*, %struct.TValue*, %struct.CallInfo*, %struct.CallInfo*, i32, i32, i16, i16, i8, i8, i32, i32, void (%struct.lua_State*, %struct.lua_Debug*)*, %struct.TValue, %struct.TValue, %struct.GCObject*, %struct.GCObject*, %struct.lua_longjmp*, i32 }
	%struct.lua_longjmp = type { %struct.lua_longjmp*, [18 x i32], i32 }
	%struct.stringtable = type { %struct.GCObject**, i32, i32 }
@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (%struct.lua_State*)* @os_clock to i8*)], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define i32 @os_clock(%struct.lua_State* nocapture %L) nounwind ssp {
entry:
	%0 = tail call i32 @"\01_clock$UNIX2003"() nounwind		; <i32> [#uses=1]
	%1 = uitofp i32 %0 to double		; <double> [#uses=1]
	%2 = fdiv double %1, 1.000000e+06		; <double> [#uses=1]
	%3 = getelementptr %struct.lua_State* %L, i32 0, i32 4		; <%struct.TValue**> [#uses=3]
	%4 = load %struct.TValue** %3, align 4		; <%struct.TValue*> [#uses=2]
	%5 = getelementptr %struct.TValue* %4, i32 0, i32 0, i32 0		; <double*> [#uses=1]
	store double %2, double* %5, align 4
	%6 = getelementptr %struct.TValue* %4, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 3, i32* %6, align 4
	%7 = load %struct.TValue** %3, align 4		; <%struct.TValue*> [#uses=1]
	%8 = getelementptr %struct.TValue* %7, i32 1		; <%struct.TValue*> [#uses=1]
	store %struct.TValue* %8, %struct.TValue** %3, align 4
	ret i32 1
}

declare i32 @"\01_clock$UNIX2003"()
