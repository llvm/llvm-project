include "../../../../include/lldb/Core/PropertiesBase.td"

let Definition = "symbollocatorsymstore", Path = "plugin.symbol-locator.symstore" in {
  def SymStoreURLs : Property<"urls", "Array">,
                     ElementType<"String">,
                     Desc<"List of local symstore directories to query after "
                          "_NT_SYMBOL_PATH">;
  def CachePath : Property<"cache", "String">,
    DefaultStringValue<"">,
    Desc<"Default cache directory for downloaded symbol files. Used when no cache is specified in _NT_SYMBOL_PATH.">;
  def TLSCertFingerprint : Property<"tls-cert-fingerprint", "String">,
    DefaultStringValue<"">,
    Desc<"SHA-256 fingerprint (lowercase hex, no separators) of the symbol server's TLS certificate. When set, LLDB will accept an HTTPS server that presents this self-signed certificate even if it is not trusted by the system certificate store (Windows only).">;
}
