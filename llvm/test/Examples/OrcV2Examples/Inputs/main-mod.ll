define i32 @main(i32 %argc, ptr %argv) {
entry:
  %and = and i32 %argc, 1
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call i32 @foo() #2
  br label %return

if.end:                                           ; preds = %entry
  %call1 = tail call i32 @bar() #2
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.end ]
  ret i32 %retval.0
}

declare i32 @foo()
declare i32 @bar()

^0 = module: (path: "main-mod.o", hash: (1466373418, 2110622332, 1230295500, 3229354382, 2004933020))
^1 = gv: (name: "foo") ; guid = 6699318081062747564
^2 = gv: (name: "main", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 22, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 1, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0), calls: ((callee: ^1), (callee: ^3))))) ; guid = 15822663052811949562
^3 = gv: (name: "bar") ; guid = 16434608426314478903
^4 = blockcount: 0
