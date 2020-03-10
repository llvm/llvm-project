extension Array where Element: Comparable {
  public func union(_ rhs: [Element]) -> [Element] {
    return [] //%self.expect('frame variable -d run -- rhs', substrs=['([Int]) rhs = 1 value'])
  }
}

var patatino = [1]
patatino.union([2])

extension Collection where Element: Equatable {
    func split<C: Collection>(separatedBy separator: C) -> [SubSequence] where C.Element == Element {
        var results = [SubSequence]() //%self.expect('frame variable -d run -- separator', substrs=['(String) separator'])
        return results
    }
}

"patatino".split(separatedBy: "p")
