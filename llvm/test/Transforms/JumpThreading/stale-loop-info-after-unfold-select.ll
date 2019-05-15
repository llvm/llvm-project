; RUN: opt -passes='require<loops>,jump-threading,verify<loops>' -S < %s

%"type1" = type { i8 }
%"type2" = type opaque

define dso_local i16* @func2(%"type1"* %this, %"type2"*) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %func1.exit, %entry
  %month.0 = phi i32 [ undef, %entry ], [ %month.0.be, %func1.exit ]
  switch i32 %month.0, label %if.end.i [
    i32 4, label %func1.exit
    i32 1, label %func1.exit
  ]

if.end.i:                                         ; preds = %while.cond
  br label %func1.exit

func1.exit:                  ; preds = %if.end.i, %while.cond, %while.cond
  %retval.0.i = phi i32 [ 9, %if.end.i ], [ 0, %while.cond ], [ 0, %while.cond ]
  %call2 = tail call signext i32 @func3(i32 signext %retval.0.i, i32 signext 1, i32 signext 3)
  %cmp = icmp slt i32 %call2, 1
  %add = add nsw i32 %call2, 2
  %month.0.be = select i1 %cmp, i32 %add, i32 %call2
  br label %while.cond
}

declare i32 @func3(i32, i32, i32)

