;;; llvm-mode.el --- Major mode for the LLVM assembler language.

;; Maintainer:  The LLVM team, http://llvm.org/
;; Version: 1.0

;;; Commentary:

;; Major mode for editing LLVM IR files.

;;; Code:

(defvar llvm-mode-syntax-table
  (let ((table (make-syntax-table)))
    (modify-syntax-entry ?% "_" table)
    (modify-syntax-entry ?. "_" table)
    (modify-syntax-entry ?\; "< " table)
    (modify-syntax-entry ?\n "> " table)
    table)
  "Syntax table used while in LLVM mode.")

(defvar llvm-font-lock-keywords
  (list
   ;; Attributes
   `(,(regexp-opt
       '("alwaysinline" "argmemonly" "allocsize" "builtin" "cold" "convergent" "dereferenceable_or_null" "hot" "inaccessiblememonly"
         "inaccessiblemem_or_argmemonly" "inalloca" "inlinehint" "jumptable" "minsize" "mustprogress" "naked" "nobuiltin"
         "nocallback" "nocf_check" "noduplicate" "nofree" "noimplicitfloat" "noinline" "nomerge" "nonlazybind" "noprofile" "noredzone" "noreturn"
         "norecurse" "nosync" "noundef" "nounwind" "null_pointer_is_valid" "optforfuzzing" "optnone" "optsize" "preallocated" "readnone" "readonly" "returns_twice"
         "shadowcallstack" "speculatable" "speculative_load_hardening" "ssp" "sspreq" "sspstrong" "safestack" "sanitize_address" "sanitize_hwaddress" "sanitize_memtag"
         "sanitize_thread" "sanitize_memory" "strictfp" "swifterror" "uwtable" "willreturn" "writeonly" "immarg") 'symbols) . font-lock-constant-face)
   ;; Variables
   '("%[-a-zA-Z$._][-a-zA-Z$._0-9]*" . font-lock-variable-name-face)
   ;; Labels
   '("[-a-zA-Z$._0-9]+:" . font-lock-variable-name-face)
   ;; Unnamed variable slots
   '("%[-]?[0-9]+" . font-lock-variable-name-face)
   ;; Types
   `(,(regexp-opt '("void" "i1" "i8" "i16" "i32" "i64" "i128" "float" "double" "type" "label" "opaque") 'symbols) . font-lock-type-face)
   ;; Integer literals
   '("\\b[-]?[0-9]+\\b" . font-lock-preprocessor-face)
   ;; Floating point constants
   '("\\b[-+]?[0-9]+.[0-9]*\\([eE][-+]?[0-9]+\\)?\\b" . font-lock-preprocessor-face)
   ;; Hex constants
   '("\\b0x[0-9A-Fa-f]+\\b" . font-lock-preprocessor-face)
   ;; Keywords
   `(,(regexp-opt
       '(;; Toplevel entities
         "declare" "define" "module" "target" "source_filename" "global" "constant" "const"
         "attributes" "uselistorder" "uselistorder_bb"
         ;; Linkage types
         "private" "internal" "weak" "weak_odr" "linkonce" "linkonce_odr" "available_externally" "appending" "common" "extern_weak" "external"
         "uninitialized" "implementation" "..."
         ;; Values
         "true" "false" "null" "undef" "zeroinitializer" "none" "c" "asm" "blockaddress"

         ;; Calling conventions
         "ccc" "fastcc" "coldcc" "webkit_jscc" "anyregcc" "preserve_mostcc" "preserve_allcc"
         "cxx_fast_tlscc" "swiftcc"

         "atomic" "volatile" "personality" "prologue" "section") 'symbols) . font-lock-keyword-face)
   ;; Arithmetic and Logical Operators
   `(,(regexp-opt '("add" "sub" "mul" "sdiv" "udiv" "urem" "srem" "and" "or" "xor"
                    "setne" "seteq" "setlt" "setgt" "setle" "setge") 'symbols) . font-lock-keyword-face)
   ;; Floating-point operators
   `(,(regexp-opt '("fadd" "fsub" "fneg" "fmul" "fdiv" "frem") 'symbols) . font-lock-keyword-face)
   ;; Special instructions
   `(,(regexp-opt '("phi" "tail" "call" "select" "to" "shl" "lshr" "ashr" "fcmp" "icmp" "va_arg" "landingpad" "freeze") 'symbols) . font-lock-keyword-face)
   ;; Control instructions
   `(,(regexp-opt '("ret" "br" "switch" "invoke" "resume" "unwind" "unreachable" "indirectbr" "callbr") 'symbols) . font-lock-keyword-face)
   ;; Memory operators
   `(,(regexp-opt '("malloc" "alloca" "free" "load" "store" "getelementptr" "fence" "cmpxchg" "atomicrmw") 'symbols) . font-lock-keyword-face)
   ;; Casts
   `(,(regexp-opt '("bitcast" "inttoptr" "ptrtoint" "trunc" "zext" "sext" "fptrunc" "fpext" "fptoui" "fptosi" "uitofp" "sitofp" "addrspacecast") 'symbols) . font-lock-keyword-face)
   ;; Vector ops
   `(,(regexp-opt '("extractelement" "insertelement" "shufflevector") 'symbols) . font-lock-keyword-face)
   ;; Aggregate ops
   `(,(regexp-opt '("extractvalue" "insertvalue") 'symbols) . font-lock-keyword-face)
   ;; Metadata types
   `(,(regexp-opt '("distinct") 'symbols) . font-lock-keyword-face)
   ;; Use-list order directives
   `(,(regexp-opt '("uselistorder" "uselistorder_bb") 'symbols) . font-lock-keyword-face))
  "Syntax highlighting for LLVM.")

;; Emacs 23 compatibility.
(defalias 'llvm-mode-prog-mode
  (if (fboundp 'prog-mode)
      'prog-mode
    'fundamental-mode))

;;;###autoload
(define-derived-mode llvm-mode llvm-mode-prog-mode "LLVM"
  "Major mode for editing LLVM source files.
\\{llvm-mode-map}
  Runs `llvm-mode-hook' on startup."
  (setq font-lock-defaults `(llvm-font-lock-keywords))
  (setq-local comment-start ";"))

;; Associate .ll files with llvm-mode
;;;###autoload
(add-to-list 'auto-mode-alist (cons "\\.ll\\'" 'llvm-mode))

(provide 'llvm-mode)

;;; llvm-mode.el ends here
