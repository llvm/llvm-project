; Simple checks of -print-changed=dot-cfg with basic blocks that have no names
;
; Note that (mostly) only the banners are checked.
;
; Simple functionality check with SimplifyCFGPass on function with unnamed basic blocks.
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -disable-verify -S -print-changed=dot-cfg -passes=simplifycfg -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 3
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-NO-NAME-SIMPLE

; Check that only the passes that change the IR are printed for unnamed basic blocks.
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -disable-verify -S -print-changed=dot-cfg -passes=simplifycfg -filter-print-funcs=g -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 3
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-NO-NAME-FUNC-FILTER

; Check that the reporting works with -print-module-scope for unnamed basic blocks
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -disable-verify -S -print-changed=dot-cfg -passes=simplifycfg -print-module-scope -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 3
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE

define void @g(i32 %0) {
  %2 = icmp sgt i32 %0, 0
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; CHECK-DOT-CFG-NO-NAME-SIMPLE: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-NO-NAME-SIMPLE-NEXT: <body><button type="button" class="collapsible">0. Initial IR (by function)</button>
; CHECK-DOT-CFG-NO-NAME-SIMPLE-NEXT: <div class="content">
; CHECK-DOT-CFG-NO-NAME-SIMPLE-NEXT:   <p>
; CHECK-DOT-CFG-NO-NAME-SIMPLE-NEXT:   <a href="diff_0_0.pdf" target="_blank">0.0. Initial IR</a><br/>
; CHECK-DOT-CFG-NO-NAME-SIMPLE-NEXT:   </p>
; CHECK-DOT-CFG-NO-NAME-SIMPLE-NEXT: </div><br/>
; CHECK-DOT-CFG-NO-NAME-SIMPLE-NEXT:   <a href="diff_1.pdf" target="_blank">1. Pass SimplifyCFGPass on g</a><br/>
; CHECK-DOT-CFG-NO-NAME-SIMPLE-NEXT:     </p></div>
; CHECK-DOT-CFG-NO-NAME-SIMPLE-NEXT:   <a>2. PassManager{{.*}} on g ignored</a><br/>
; CHECK-DOT-CFG-NO-NAME-SIMPLE-NEXT:   <a>3. ModuleToFunctionPassAdaptor on [module] ignored</a><br/>
; CHECK-DOT-CFG-NO-NAME-SIMPLE-NEXT:   <a>4. PrintModulePass on [module] ignored</a><br/>
; CHECK-DOT-CFG-NO-NAME-SIMPLE-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER-NEXT: <body><button type="button" class="collapsible">0. Initial IR (by function)</button>
; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER-NEXT: <div class="content">
; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER-NEXT:   <p>
; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER-NEXT:   <a href="diff_0_0.pdf" target="_blank">0.0. Initial IR</a><br/>
; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER-NEXT:   </p>
; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER-NEXT: </div><br/>
; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER-NEXT:   <a href="diff_1.pdf" target="_blank">1. Pass SimplifyCFGPass on g</a><br/>
; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER-NEXT:     </p></div>
; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER-NEXT:   <a>2. PassManager{{.*}} on g ignored</a><br/>
; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER-NEXT:   <a>3. ModuleToFunctionPassAdaptor on [module] ignored</a><br/>
; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER-NEXT:   <a>4. PrintModulePass on [module] ignored</a><br/>
; CHECK-DOT-CFG-NO-NAME-FUNC-FILTER-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE-NEXT: <body><button type="button" class="collapsible">0. Initial IR (by function)</button>
; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE-NEXT: <div class="content">
; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE-NEXT:   <p>
; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE-NEXT:   <a href="diff_0_0.pdf" target="_blank">0.0. Initial IR</a><br/>
; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE-NEXT:   </p>
; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE-NEXT: </div><br/>
; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE-NEXT:   <a href="diff_1.pdf" target="_blank">1. Pass SimplifyCFGPass on g</a><br/>
; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE-NEXT:     </p></div>
; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE-NEXT:   <a>2. PassManager{{.*}} on g ignored</a><br/>
; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE-NEXT:   <a>3. ModuleToFunctionPassAdaptor on [module] ignored</a><br/>
; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE-NEXT:   <a>4. PrintModulePass on [module] ignored</a><br/>
; CHECK-DOT-CFG-NO-NAME-PRINT-MOD-SCOPE-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>
