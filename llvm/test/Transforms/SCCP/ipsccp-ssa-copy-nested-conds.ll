; RUN: opt < %s -ipsccp -S | FileCheck %s
; RUN: opt < %s -passes=ipsccp -S | FileCheck %s

; Test for PR39772
; CHECK-LABEL: cleanup:
; CHECK-NEXT:   %retval.0 = phi i32 [ 0, %if.then ], [ %add, %if.then7 ], [ %add8, %if.else ]


%struct.Node = type { %struct.Node*, %struct.Node*, i32 }

define i32 @check(%struct.Node* %node) {
entry:
  %cmp = icmp eq %struct.Node* %node, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %cleanup

if.end:                                           ; preds = %entry
  %left = getelementptr inbounds %struct.Node, %struct.Node* %node, i32 0, i32 0
  %0 = load %struct.Node*, %struct.Node** %left
  %call = call i32 @check(%struct.Node* %0)
  %right = getelementptr inbounds %struct.Node, %struct.Node* %node, i32 0, i32 1
  %1 = load %struct.Node*, %struct.Node** %right
  %call1 = call i32 @check(%struct.Node* %1)
  %2 = load %struct.Node*, %struct.Node** %right
  %height = getelementptr inbounds %struct.Node, %struct.Node* %2, i32 0, i32 2
  %3 = load i32, i32* %height
  %cmp3 = icmp ne i32 %3, %call1
  br i1 %cmp3, label %if.then4, label %if.end5

if.then4:                                         ; preds = %if.end
  unreachable

if.end5:                                          ; preds = %if.end
  %cmp6 = icmp sgt i32 %call, %call1
  br i1 %cmp6, label %if.then7, label %if.else

if.then7:                                         ; preds = %if.end5
  %add = add nsw i32 %call, 1
  br label %cleanup

if.else:                                          ; preds = %if.end5
  %add8 = add nsw i32 %call1, 1
  br label %cleanup

cleanup:                                          ; preds = %if.else, %if.then7, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ %add, %if.then7 ], [ %add8, %if.else ]
  ret i32 %retval.0
}
