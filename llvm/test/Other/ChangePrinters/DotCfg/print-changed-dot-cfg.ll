; Simple checks of -print-changed=dot-cfg
;
; Note that (mostly) only the banners are checked.
;
; Simple functionality check.
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -disable-verify -S -print-changed=dot-cfg -passes=instsimplify -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 5
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-SIMPLE
;
; Check that only the passes that change the IR are printed and that the
; others (including g) are filtered out.
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -disable-verify -S -print-changed=dot-cfg -passes=instsimplify -filter-print-funcs=f  -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 3
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-FUNC-FILTER
;
; Check that the reporting of IRs respects is not affected by
; -print-module-scope
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -disable-verify -S -print-changed=dot-cfg -passes=instsimplify -print-module-scope -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 5
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-PRINT-MOD-SCOPE
;
; Check that reporting of multiple functions happens
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -disable-verify -S -print-changed=dot-cfg -passes=instsimplify -filter-print-funcs="f,g" -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 5
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-FILTER-MULT-FUNC
;
; Check that the reporting of IRs respects -filter-passes
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -disable-verify -S -print-changed=dot-cfg -passes="instsimplify,no-op-function" -filter-passes="no-op-function" -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 3
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-FILTER-PASSES
;
; Check that the reporting of IRs respects -filter-passes with multiple passes
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -disable-verify -S -print-changed=dot-cfg -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 5
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-FILTER-MULT-PASSES
;
; Check that the reporting of IRs respects both -filter-passes and -filter-print-funcs
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -disable-verify -S -print-changed=dot-cfg -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" -filter-print-funcs=f -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 3
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-FILTER-FUNC-PASSES
;
; Check that repeated passes that change the IR are printed and that the
; others (including g) are filtered out.  Note that only the first time
; instsimplify is run on f will result in changes
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -disable-verify -S -print-changed=dot-cfg -passes="instsimplify,instsimplify" -filter-print-funcs=f  -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 3
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC
;
; Simple checks of -print-changed=dot-cfg-quiet
;
; Note that (mostly) only the banners are checked.
;
; Simple functionality check.
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -S -print-changed=dot-cfg-quiet -passes=instsimplify -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 3
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-QUIET-SIMPLE --allow-empty
;
; Check that only the passes that change the IR are printed and that the
; others (including g) are filtered out.
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -S -print-changed=dot-cfg-quiet -passes=instsimplify -filter-print-funcs=f  -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 2
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-QUIET-FUNC-FILTER
;
; Check that the reporting of IRs respects is not affected by
; -print-module-scope
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -S -print-changed=dot-cfg-quiet -passes=instsimplify -print-module-scope -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 3
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-QUIET-PRINT-MOD-SCOPE
;
; Check that reporting of multiple functions happens
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -S -print-changed=dot-cfg-quiet -passes=instsimplify -filter-print-funcs="f,g" -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 3
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-QUIET-FILTER-MULT-FUNC
;
; Check that the reporting of IRs respects -filter-passes
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -S -print-changed=dot-cfg-quiet -passes="instsimplify,no-op-function" -filter-passes="no-op-function" -dot-cfg-dir=%t < %s -o /dev/null
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-QUIET-FILTER-PASSES-NONE --allow-empty
;
; Check that the reporting of IRs respects -filter-passes with multiple passes
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -S -print-changed=dot-cfg-quiet -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 3
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-QUIET-FILTER-MULT-PASSES
;
; Check that the reporting of IRs respects both -filter-passes and -filter-print-funcs
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -S -print-changed=dot-cfg-quiet -passes="instsimplify,no-op-function" -filter-passes="no-op-function,instsimplify" -filter-print-funcs=f -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 2
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-QUIET-FILTER-FUNC-PASSES
;
; Check that repeated passes that change the IR are printed and that the
; others (including g) are filtered out.  Note that only the first time
; instsimplify is run on f will result in changes
; RUN: rm -rf %t && mkdir -p %t
; RUN: opt -S -print-changed=dot-cfg-quiet -passes="instsimplify,instsimplify" -filter-print-funcs=f  -dot-cfg-dir=%t < %s -o /dev/null
; RUN: ls %t/*.pdf %t/passes.html | count 2
; RUN: FileCheck %s -input-file=%t/passes.html --check-prefix=CHECK-DOT-CFG-QUIET-MULT-PASSES-FILTER-FUNC

define i32 @g() {
entry:
  %a = add i32 2, 3
  ret i32 %a
}

define i32 @f() {
entry:
  %a = add i32 2, 3
  ret i32 %a
}

; CHECK-DOT-CFG-SIMPLE-FILES: passes.html diff_0.pdf diff_1.pdf diff_3.pdf
; CHECK-DOT-CFG-SIMPLE: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-SIMPLE-NEXT: <body><button type="button" class="collapsible">0. Initial IR (by function)</button>
; CHECK-DOT-CFG-SIMPLE-NEXT: <div class="content">
; CHECK-DOT-CFG-SIMPLE-NEXT:   <p>
; CHECK-DOT-CFG-SIMPLE-NEXT:   <a href="diff_0_0.pdf" target="_blank">0.0. Initial IR</a><br/>
; CHECK-DOT-CFG-SIMPLE-NEXT:   <a href="diff_0_1.pdf" target="_blank">0.1. Initial IR</a><br/>
; CHECK-DOT-CFG-SIMPLE-NEXT:   </p>
; CHECK-DOT-CFG-SIMPLE-NEXT: </div><br/>
; CHECK-DOT-CFG-SIMPLE-NEXT:   <a href="diff_1.pdf" target="_blank">1. Pass InstSimplifyPass on g</a><br/>
; CHECK-DOT-CFG-SIMPLE-NEXT:     </p></div>
; CHECK-DOT-CFG-SIMPLE-NEXT:   <a>2. PassManager{{.*}} on g ignored</a><br/>
; CHECK-DOT-CFG-SIMPLE-NEXT:   <a href="diff_3.pdf" target="_blank">3. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-SIMPLE-NEXT:     </p></div>
; CHECK-DOT-CFG-SIMPLE-NEXT:   <a>4. PassManager{{.*}} on f ignored</a><br/>
; CHECK-DOT-CFG-SIMPLE-NEXT:   <a>5. ModuleToFunctionPassAdaptor on [module] ignored</a><br/>
; CHECK-DOT-CFG-SIMPLE-NEXT:   <a>6. Pass PrintModulePass on [module] omitted because no change</a><br/>
; CHECK-DOT-CFG-SIMPLE-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-FUNC-FILTER: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT: <body><button type="button" class="collapsible">0. Initial IR (by function)</button>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT: <div class="content">
; CHECK-DOT-CFG-FUNC-FILTER-NEXT:   <p>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT:   <a href="diff_0_0.pdf" target="_blank">0.0. Initial IR</a><br/>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT:   </p>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT: </div><br/>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT: <a>1. Pass InstSimplifyPass on g filtered out</a><br/>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT:   <a>2. PassManager{{.*}} on g ignored</a><br/>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT:   <a href="diff_3.pdf" target="_blank">3. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT:     </p></div>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT:   <a>4. PassManager{{.*}} on f ignored</a><br/>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT:   <a>5. ModuleToFunctionPassAdaptor on [module] ignored</a><br/>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT:   <a>6. Pass PrintModulePass on [module] omitted because no change</a><br/>
; CHECK-DOT-CFG-FUNC-FILTER-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-PRINT-MOD-SCOPE: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT: <body><button type="button" class="collapsible">0. Initial IR (by function)</button>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT: <div class="content">
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT:   <p>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT:   <a href="diff_0_0.pdf" target="_blank">0.0. Initial IR</a><br/>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT:   <a href="diff_0_1.pdf" target="_blank">0.1. Initial IR</a><br/>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT:   </p>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT: </div><br/>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT:   <a href="diff_1.pdf" target="_blank">1. Pass InstSimplifyPass on g</a><br/>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT:     </p></div>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT:   <a>2. PassManager{{.*}} on g ignored</a><br/>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT:   <a href="diff_3.pdf" target="_blank">3. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT:     </p></div>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT:   <a>4. PassManager{{.*}} on f ignored</a><br/>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT:   <a>5. ModuleToFunctionPassAdaptor on [module] ignored</a><br/>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT:   <a>6. Pass PrintModulePass on [module] omitted because no change</a><br/>
; CHECK-DOT-CFG-PRINT-MOD-SCOPE-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-FILTER-MULT-FUNC: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT: <body><button type="button" class="collapsible">0. Initial IR (by function)</button>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT: <div class="content">
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT:   <p>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT:   <a href="diff_0_0.pdf" target="_blank">0.0. Initial IR</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT:   <a href="diff_0_1.pdf" target="_blank">0.1. Initial IR</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT:   </p>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT: </div><br/>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT:   <a href="diff_1.pdf" target="_blank">1. Pass InstSimplifyPass on g</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT:     </p></div>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT:   <a>2. PassManager{{.*}} on g ignored</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT:   <a href="diff_3.pdf" target="_blank">3. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT:     </p></div>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT:   <a>4. PassManager{{.*}} on f ignored</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT:   <a>5. ModuleToFunctionPassAdaptor on [module] ignored</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT:   <a>6. Pass PrintModulePass on [module] omitted because no change</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-FUNC-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-FILTER-PASSES: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT: <body><button type="button" class="collapsible">0. Initial IR (by function)</button>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT: <div class="content">
; CHECK-DOT-CFG-FILTER-PASSES-NEXT:   <p>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT:   <a href="diff_0_0.pdf" target="_blank">0.0. Initial IR</a><br/>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT:   <a href="diff_0_1.pdf" target="_blank">0.1. Initial IR</a><br/>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT:   </p>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT: </div><br/>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT:   <a>1. Pass InstSimplifyPass on g filtered out</a><br/>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT:   <a>2. Pass NoOpFunctionPass on g omitted because no change</a><br/>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT:   <a>3. PassManager{{.*}} on g ignored</a><br/>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT:   <a>4. Pass InstSimplifyPass on f filtered out</a><br/>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT:   <a>5. Pass NoOpFunctionPass on f omitted because no change</a><br/>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT:   <a>6. PassManager{{.*}} on f ignored</a><br/>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT:   <a>7. ModuleToFunctionPassAdaptor on [module] ignored</a><br/>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT:   <a>8. Pass PrintModulePass on [module] filtered out</a><br/>
; CHECK-DOT-CFG-FILTER-PASSES-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>



; CHECK-DOT-CFG-FILTER-MULT-PASSES: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT: <button type="button" class="collapsible">0. Initial IR (by function)</button>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT: <div class="content">
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:   <p>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:   <a href="diff_0_0.pdf" target="_blank">0.0. Initial IR</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:   <a href="diff_0_1.pdf" target="_blank">0.1. Initial IR</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:   </p>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT: </div><br/>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:   <a href="diff_1.pdf" target="_blank">1. Pass InstSimplifyPass on g</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:     </p></div>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:   <a>2. Pass NoOpFunctionPass on g omitted because no change</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:   <a>3. PassManager{{.*}} on g ignored</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:   <a href="diff_4.pdf" target="_blank">4. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:     </p></div>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:   <a>5. Pass NoOpFunctionPass on f omitted because no change</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:   <a>6. PassManager{{.*}} on f ignored</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:   <a>7. ModuleToFunctionPassAdaptor on [module] ignored</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT:   <a>8. Pass PrintModulePass on [module] filtered out</a><br/>
; CHECK-DOT-CFG-FILTER-MULT-PASSES-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-FILTER-FUNC-PASSES: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT: <button type="button" class="collapsible">0. Initial IR (by function)</button>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT: <div class="content">
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT:   <p>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT:   <a href="diff_0_0.pdf" target="_blank">0.0. Initial IR</a><br/>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT:   </p>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT: </div><br/>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT:   <a>1. Pass InstSimplifyPass on g filtered out</a><br/>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT:   <a>2. Pass NoOpFunctionPass on g filtered out</a><br/>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT:   <a>3. PassManager{{.*}} on g ignored</a><br/>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT:   <a href="diff_4.pdf" target="_blank">4. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT:     </p></div>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT:   <a>5. Pass NoOpFunctionPass on f omitted because no change</a><br/>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT:   <a>6. PassManager{{.*}} on f ignored</a><br/>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT:   <a>7. ModuleToFunctionPassAdaptor on [module] ignored</a><br/>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT:   <a>8. Pass PrintModulePass on [module] filtered out</a><br/>
; CHECK-DOT-CFG-FILTER-FUNC-PASSES-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>


; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT: <button type="button" class="collapsible">0. Initial IR (by function)</button>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT: <div class="content">
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT:   <p>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT:   <a href="diff_0_0.pdf" target="_blank">0.0. Initial IR</a><br/>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT:   </p>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT: </div><br/>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT:   <a>1. Pass InstSimplifyPass on g filtered out</a><br/>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT:   <a>2. Pass InstSimplifyPass on g filtered out</a><br/>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT:   <a>3. PassManager{{.*}} on g ignored</a><br/>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT:   <a href="diff_4.pdf" target="_blank">4. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT:     </p></div>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT:   <a>5. Pass InstSimplifyPass on f omitted because no change</a><br/>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT:   <a>6. PassManager{{.*}} on f ignored</a><br/>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT:   <a>7. ModuleToFunctionPassAdaptor on [module] ignored</a><br/>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT:   <a>8. Pass PrintModulePass on [module] omitted because no change</a><br/>
; CHECK-DOT-CFG-MULT-PASSES-FILTER-FUNC-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-QUIET-SIMPLE: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-QUIET-SIMPLE-NEXT: <body>  <a href="diff_0.pdf" target="_blank">0. Pass InstSimplifyPass on g</a><br/>
; CHECK-DOT-CFG-QUIET-SIMPLE-NEXT:     </p></div>
; CHECK-DOT-CFG-QUIET-SIMPLE-NEXT:   <a href="diff_1.pdf" target="_blank">1. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-QUIET-SIMPLE-NEXT:     </p></div>
; CHECK-DOT-CFG-QUIET-SIMPLE-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-QUIET-FUNC-FILTER: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-QUIET-FUNC-FILTER-NEXT: <body>  <a href="diff_0.pdf" target="_blank">0. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-QUIET-FUNC-FILTER-NEXT:     </p></div>
; CHECK-DOT-CFG-QUIET-FUNC-FILTER-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-QUIET-PRINT-MOD-SCOPE: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-QUIET-PRINT-MOD-SCOPE-NEXT: <body>  <a href="diff_0.pdf" target="_blank">0. Pass InstSimplifyPass on g</a><br/>
; CHECK-DOT-CFG-QUIET-PRINT-MOD-SCOPE-NEXT:     </p></div>
; CHECK-DOT-CFG-QUIET-PRINT-MOD-SCOPE-NEXT:   <a href="diff_1.pdf" target="_blank">1. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-QUIET-PRINT-MOD-SCOPE-NEXT:     </p></div>
; CHECK-DOT-CFG-QUIET-PRINT-MOD-SCOPE-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-QUIET-FILTER-MULT-FUNC: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-QUIET-FILTER-MULT-FUNC-NEXT: <body>  <a href="diff_0.pdf" target="_blank">0. Pass InstSimplifyPass on g</a><br/>
; CHECK-DOT-CFG-QUIET-FILTER-MULT-FUNC-NEXT:     </p></div>
; CHECK-DOT-CFG-QUIET-FILTER-MULT-FUNC-NEXT:   <a href="diff_1.pdf" target="_blank">1. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-QUIET-FILTER-MULT-FUNC-NEXT:     </p></div>
; CHECK-DOT-CFG-QUIET-FILTER-MULT-FUNC-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-QUIET-FILTER-PASSES-NONE: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-QUIET-FILTER-PASSES-NONE-NEXT: <body><script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-QUIET-FILTER-MULT-PASSES: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-QUIET-FILTER-MULT-PASSES-NEXT: <body>  <a href="diff_0.pdf" target="_blank">0. Pass InstSimplifyPass on g</a><br/>
; CHECK-DOT-CFG-QUIET-FILTER-MULT-PASSES-NEXT:     </p></div>
; CHECK-DOT-CFG-QUIET-FILTER-MULT-PASSES-NEXT:   <a href="diff_1.pdf" target="_blank">1. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-QUIET-FILTER-MULT-PASSES-NEXT:     </p></div>
; CHECK-DOT-CFG-QUIET-FILTER-MULT-PASSES-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-QUIET-FILTER-FUNC-PASSES: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-QUIET-FILTER-FUNC-PASSES-NEXT: <body>  <a href="diff_0.pdf" target="_blank">0. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-QUIET-FILTER-FUNC-PASSES-NEXT:     </p></div>
; CHECK-DOT-CFG-QUIET-FILTER-FUNC-PASSES-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>

; CHECK-DOT-CFG-QUIET-MULT-PASSES-FILTER-FUNC: <!doctype html><html><head><style>.collapsible { background-color: #777; color: white; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px;} .active, .collapsible:hover { background-color: #555;} .content { padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1;}</style><title>passes.html</title></head>
; CHECK-DOT-CFG-QUIET-MULT-PASSES-FILTER-FUNC-NEXT: <body>  <a href="diff_0.pdf" target="_blank">0. Pass InstSimplifyPass on f</a><br/>
; CHECK-DOT-CFG-QUIET-MULT-PASSES-FILTER-FUNC-NEXT:     </p></div>
; CHECK-DOT-CFG-QUIET-MULT-PASSES-FILTER-FUNC-NEXT: <script>var coll = document.getElementsByClassName("collapsible");var i;for (i = 0; i < coll.length; i++) {coll[i].addEventListener("click", function() { this.classList.toggle("active"); var content = this.nextElementSibling; if (content.style.display === "block"){ content.style.display = "none"; } else { content.style.display= "block"; } }); }</script></body></html>
