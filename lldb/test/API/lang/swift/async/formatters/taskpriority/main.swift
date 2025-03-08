@main struct Entry {
  static func main() {
    let high = TaskPriority.high
    let medium = TaskPriority.medium
    let low = TaskPriority.low
    let background = TaskPriority.background
    let custom = TaskPriority(rawValue: 15)
    print("break here")
  }
}
