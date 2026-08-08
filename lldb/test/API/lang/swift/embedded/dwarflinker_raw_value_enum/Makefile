SWIFT_SOURCES := main.swift

# The DWARFLinker truncation this test covers only reproduces at -O. At -Onone
# the enumeration type is reached first through a local variable's DW_AT_type,
# which marks its enumerators Keep before the parent walk can short-circuit.
SWIFTFLAGS_EXTRAS := -O

include Makefile.rules
