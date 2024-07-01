;;; tblgen-lsp.el --- Description -*- lexical-binding: t; -*-
;;
;; Package-Requires: ((emacs "24.3"))
;;
;; This file is not part of GNU Emacs.
;;
;;; Commentary:
;;  LSP clinet to use with `tablegen-mode' that uses `tblgen-lsp-server' or any
;;  user made compatible server.
;;
;;
;;; Code:


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
  "Args of LSP client for TableGen "
  :group 'lsp-tblgen
  :risky t
  :type 'file)

(defun lsp-tblgen-setup ()
  "Setup the LSP client for TableGen."
  (add-to-list 'lsp-language-id-configuration '(tablegen-mode . "tablegen"))

  (lsp-register-client
   (make-lsp-client
    :new-connection (lsp-stdio-connection (lambda () (cons lsp-tblgen-server-executable lsp-tblgen-server-args))); (concat "--tablegen-compilation-database=" lsp-tblgen-compilation-database-location) )))
    :activation-fn (lsp-activate-on "tablegen")
    :priority -1
    :server-id 'tblgen-lsp-server)))

(provide 'tblgen-lsp)
;;; tblgen-lsp.el ends here
