import Lib

// A cross-module extension of Lib.Root: in the DWARF for this compile unit the
// getter's decl context is Lib (module) -> Root (struct) -> doubled.get.
extension Root {
  var doubled: Int {
    let factor = 7
    return value * factor // break here
  }
}

print(Root(value: 21).doubled)
