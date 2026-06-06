#------------------------------------------------------------------------------
# Load-acquire/store-release
#------------------------------------------------------------------------------

ldapur     w7, [x24]
ldapur     x20, [x13]
ldapurb    w13, [x17]
ldapurh    w3, [x22]
ldapursb   w7, [x8]
ldapursb   x29, [x7]
ldapursh   w17, [x19]
ldapursh   x3, [x3]
ldapursw   x3, [x18]
stlur      w3, [x27]
stlur      x23, [x25]
stlurb     w30, [x17]
stlurh     w9, [x29]
