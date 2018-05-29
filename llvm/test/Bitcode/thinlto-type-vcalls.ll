; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: llvm-dis -o - %t.o | FileCheck %s --check-prefix=DIS
; RUN: llvm-lto -thinlto -o %t2 %t.o
; RUN: llvm-bcanalyzer -dump %t2.thinlto.bc | FileCheck --check-prefix=COMBINED %s
; RUN: llvm-dis -o - %t2.thinlto.bc | FileCheck %s --check-prefix=COMBINED-DIS

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; COMBINED:      <TYPE_TEST_ASSUME_VCALLS op0=6699318081062747564 op1=16/>
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <TYPE_CHECKED_LOAD_VCALLS op0=6699318081062747564 op1=16/>
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <TYPE_TEST_ASSUME_VCALLS op0=6699318081062747564 op1=24 op2=-2012135647395072713 op3=32/>
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <TYPE_TEST_ASSUME_CONST_VCALL op0=6699318081062747564 op1=16 op2=42/>
; COMBINED-NEXT: <TYPE_TEST_ASSUME_CONST_VCALL op0=6699318081062747564 op1=24 op2=43/>
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <TYPE_CHECKED_LOAD_CONST_VCALL op0=6699318081062747564 op1=16 op2=42/>
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <TYPE_TESTS op0=7546896869197086323/>
; COMBINED-NEXT: <COMBINED

; CHECK: <TYPE_TEST_ASSUME_VCALLS op0=6699318081062747564 op1=16/>
define void @f1([3 x i8*]* %vtable) {
  %vtablei8 = bitcast [3 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"foo")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x i8*], [3 x i8*]* %vtable, i32 0, i32 2
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to void (i8*, i32)*
  call void %fptr_casted(i8* null, i32 undef)
  ret void
}

; CHECK: <TYPE_TEST_ASSUME_VCALLS op0=6699318081062747564 op1=24 op2=-2012135647395072713 op3=32/>
define void @f2([3 x i8*]* %vtable, [3 x i8*]* %vtable2) {
  %vtablei8 = bitcast [3 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"foo")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x i8*], [3 x i8*]* %vtable, i32 0, i32 3
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to void (i8*, i32)*
  call void %fptr_casted(i8* null, i32 undef)

  %vtablei82 = bitcast [3 x i8*]* %vtable2 to i8*
  %p2 = call i1 @llvm.type.test(i8* %vtablei82, metadata !"bar")
  call void @llvm.assume(i1 %p2)
  %fptrptr2 = getelementptr [3 x i8*], [3 x i8*]* %vtable2, i32 0, i32 4
  %fptr2 = load i8*, i8** %fptrptr2
  %fptr_casted2 = bitcast i8* %fptr2 to void (i8*, i128)*
  call void %fptr_casted2(i8* null, i128 0)

  ret void
}

; CHECK: <TYPE_CHECKED_LOAD_VCALLS op0=6699318081062747564 op1=16/>
define void @f3(i8* %vtable) {
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtable, i32 16, metadata !"foo")
  %fptr = extractvalue {i8*, i1} %pair, 0
  %fptr_casted = bitcast i8* %fptr to void (i8*, i32)*
  call void %fptr_casted(i8* null, i32 undef)
  ret void
}

; CHECK: <TYPE_TEST_ASSUME_CONST_VCALL op0=6699318081062747564 op1=16 op2=42/>
; CHECK-NEXT: <TYPE_TEST_ASSUME_CONST_VCALL op0=6699318081062747564 op1=24 op2=43/>
define void @f4([3 x i8*]* %vtable, [3 x i8*]* %vtable2) {
  %vtablei8 = bitcast [3 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"foo")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x i8*], [3 x i8*]* %vtable, i32 0, i32 2
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to void (i8*, i32)*
  call void %fptr_casted(i8* null, i32 42)

  %vtablei82 = bitcast [3 x i8*]* %vtable2 to i8*
  %p2 = call i1 @llvm.type.test(i8* %vtablei82, metadata !"foo")
  call void @llvm.assume(i1 %p2)
  %fptrptr2 = getelementptr [3 x i8*], [3 x i8*]* %vtable2, i32 0, i32 3
  %fptr2 = load i8*, i8** %fptrptr2
  %fptr_casted2 = bitcast i8* %fptr2 to void (i8*, i32)*
  call void %fptr_casted2(i8* null, i32 43)
  ret void
}

; CHECK: <TYPE_CHECKED_LOAD_CONST_VCALL op0=6699318081062747564 op1=16 op2=42/>
define void @f5(i8* %vtable) {
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtable, i32 16, metadata !"foo")
  %fptr = extractvalue {i8*, i1} %pair, 0
  %fptr_casted = bitcast i8* %fptr to void (i8*, i32)*
  call void %fptr_casted(i8* null, i32 42)
  ret void
}

; CHECK-NOT: <TYPE_CHECKED_LOAD_CONST_VCALL op0=7546896869197086323
; CHECK: <TYPE_TESTS op0=7546896869197086323/>
; CHECK-NOT: <TYPE_CHECKED_LOAD_CONST_VCALL op0=7546896869197086323
define {i8*, i1} @f6(i8* %vtable) {
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtable, i32 16, metadata !"baz")
  ret {i8*, i1} %pair
}

declare i1 @llvm.type.test(i8*, metadata) nounwind readnone
declare void @llvm.assume(i1)
declare {i8*, i1} @llvm.type.checked.load(i8*, i32, metadata)

; DIS: ^0 = module: (path: "{{.*}}thinlto-type-vcalls.ll.tmp.o", hash: (0, 0, 0, 0, 0))
; DIS: ^1 = gv: (name: "llvm.type.test") ; guid = 608142985856744218
; DIS: ^2 = gv: (name: "f1", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 8, typeIdInfo: (typeTestAssumeVCalls: (vFuncId: (guid: 6699318081062747564, offset: 16)))))) ; guid = 2072045998141807037
; DIS: ^3 = gv: (name: "f3", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 5, typeIdInfo: (typeCheckedLoadVCalls: (vFuncId: (guid: 6699318081062747564, offset: 16)))))) ; guid = 4197650231481825559
; DIS: ^4 = gv: (name: "llvm.type.checked.load") ; guid = 5568222536364573403
; DIS: ^5 = gv: (name: "llvm.assume") ; guid = 6385187066495850096
; DIS: ^6 = gv: (name: "f2", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 15, typeIdInfo: (typeTestAssumeVCalls: (vFuncId: (guid: 6699318081062747564, offset: 24), vFuncId: (guid: 16434608426314478903, offset: 32)))))) ; guid = 8471399308421654326
; DIS: ^7 = gv: (name: "f4", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 15, typeIdInfo: (typeTestAssumeConstVCalls: (vFuncId: (guid: 6699318081062747564, offset: 16), args: (42), vFuncId: (guid: 6699318081062747564, offset: 24), args: (43)))))) ; guid = 10064745020953272174
; DIS: ^8 = gv: (name: "f5", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 5, typeIdInfo: (typeCheckedLoadConstVCalls: (vFuncId: (guid: 6699318081062747564, offset: 16), args: (42)))))) ; guid = 11686717102184386164
; DIS: ^9 = gv: (name: "f6", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 2, typeIdInfo: (typeTests: (7546896869197086323))))) ; guid = 11834966808443348068

; COMBINED-DIS: ^0 = module: (path: "{{.*}}thinlto-type-vcalls.ll.tmp.o", hash: (0, 0, 0, 0, 0))
; COMBINED-DIS: ^1 = gv: (guid: 2072045998141807037, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 8, typeIdInfo: (typeTestAssumeVCalls: (vFuncId: (guid: 6699318081062747564, offset: 16))))))
; COMBINED-DIS: ^2 = gv: (guid: 4197650231481825559, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 5, typeIdInfo: (typeCheckedLoadVCalls: (vFuncId: (guid: 6699318081062747564, offset: 16))))))
; COMBINED-DIS: ^3 = gv: (guid: 8471399308421654326, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 15, typeIdInfo: (typeTestAssumeVCalls: (vFuncId: (guid: 6699318081062747564, offset: 24), vFuncId: (guid: 16434608426314478903, offset: 32))))))
; COMBINED-DIS: ^4 = gv: (guid: 10064745020953272174, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 15, typeIdInfo: (typeTestAssumeConstVCalls: (vFuncId: (guid: 6699318081062747564, offset: 16), args: (42), vFuncId: (guid: 6699318081062747564, offset: 24), args: (43))))))
; COMBINED-DIS: ^5 = gv: (guid: 11686717102184386164, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 5, typeIdInfo: (typeCheckedLoadConstVCalls: (vFuncId: (guid: 6699318081062747564, offset: 16), args: (42))))))
; COMBINED-DIS: ^6 = gv: (guid: 11834966808443348068, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 2, typeIdInfo: (typeTests: (7546896869197086323)))))
