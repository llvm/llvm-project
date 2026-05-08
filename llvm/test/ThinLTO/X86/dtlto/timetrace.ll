; Test that the LLD produces expected time trace output for DTLTO.

RUN: rm -rf %t && split-file %s %t && cd %t

; Generate bitcode files with summary.
RUN: opt -thinlto-bc t1.ll -o t1.bc
RUN: opt -thinlto-bc t2.ll -o t2.bc

; Generate fake object files for mock.py to return.
RUN: touch t1.o t2.o

; Perform DTLTO and generate a time trace. mock.py does not do any compilation,
; instead it simply writes the contents of the object files supplied on the
; command line into the output object files in job order.
RUN: llvm-lto2 run t1.bc t2.bc -o t.o \
RUN:   -dtlto-distributor=%python \
RUN:   -dtlto-distributor-arg=%llvm_src_root/utils/dtlto/mock.py,t1.o,t2.o \
RUN:   --time-trace --time-trace-granularity=0 --time-trace-file=%t.json \
RUN:   -r=t1.bc,t1,px \
RUN:   -r=t2.bc,t2,px
RUN: %python filter_order_and_pprint.py %t.json | FileCheck %s

## Check that DTLTO events are recorded.
CHECK-NOT:  "name"
CHECK:      "name": "Add DTLTO files to the link"
CHECK-SAME:   "pid": [[#PID:]],
CHECK-NEXT: "name": "Emit DTLTO JSON"
CHECK-NEXT: "name": "Emit individual index for DTLTO"
CHECK-SAME:   t1.1.[[#PID]].native.o.thinlto.bc"
CHECK-NEXT: "name": "Emit individual index for DTLTO"
CHECK-SAME:   t2.2.[[#PID]].native.o.thinlto.bc"
CHECK-NEXT: "name": "Execute DTLTO distributor", "{{.*}}"
CHECK-NEXT: "name": "Remove DTLTO temporary files"
CHECK-NEXT: "name": "Total Add DTLTO files to the link"
CHECK-SAME:   "count": 1,
CHECK-NEXT: "name": "Total Emit DTLTO JSON"
CHECK-SAME:   "count": 1,
CHECK-NEXT: "name": "Total Emit individual index for DTLTO"
CHECK-SAME:   "count": 2,
CHECK-NEXT: "name": "Total Execute DTLTO distributor"
CHECK-SAME:   "count": 1,
CHECK-NEXT: "name": "Total Remove DTLTO temporary files"
CHECK-SAME:   "count": 1,
CHECK-NOT:  "name"

;--- t1.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @t1() {
  ret void
}

;--- t2.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @t2() {
  ret void
}

#--- filter_order_and_pprint.py
import json, sys

data = json.load(open(sys.argv[1], "r", encoding="utf-8"))

# Get DTLTO events.
events = [e for e in data["traceEvents"] if "DTLTO" in e["name"]]
events.sort(key=lambda e: (e["name"], str(e.get("args", {}).get("detail", ""))))

# Print an event per line. Ensure 'name' is the first key.
for ev in events:
    name = ev.pop("name")
    print(json.dumps({"name": name, **ev}))

