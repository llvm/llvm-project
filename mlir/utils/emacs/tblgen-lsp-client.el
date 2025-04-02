;;; tblgen-lsp-client.el --- Description -*- lexical-binding: t; -*-
;;
;; Package-Requires: ((emacs "24.3"))
;;
;; This file is not part of GNU Emacs.
;;
;;; Commentary:
;;  LSP client to use with `tablegen-mode' that uses `tblgen-lsp-server' or any
;;  user made compatible server.
;;
;;
;;; Code:
(require 'lsp-mode)

(defgroup lsp-tblgen nil
  "LSP support for Tablegen."
  :group 'lsp-mode
  :link '(url-link "https://mlir.llvm.org/docs/Tools/MLIRLSP/"))

(defcustom lsp-tblgen-server-executable "tblgen-lsp-server"
  "Command to start the mlir language server."
  :group 'lsp-tblgen
  :risky t
  :type 'file)


(defcustom lsp-tblgen-server-args ""
  "Args of LSP client for TableGen, for example '--tablegen-compilation-database=.../build/tablegen_compile_commands.yml'"
  :group 'lsp-tblgen
  :risky t
  :type 'file)

(defun lsp-tblgen-setup ()
  "Setup the LSP client for TableGen."
  (add-to-list 'lsp-language-id-configuration '(tablegen-mode . "tablegen"))

  (lsp-register-client
   (make-lsp-client
    :new-connection (lsp-stdio-connection (lambda () (cons lsp-tblgen-server-executable (split-string-shell-command lsp-tblgen-server-args))))
    :activation-fn (lsp-activate-on "tablegen")
    :priority -1
    :server-id 'tblgen-lsp)))

(provide 'tblgen-lsp)
;;; tblgen-lsp-client.el ends here
