enum ArgumentParseError: Error {
  case failure(String)
  case unexpectedArgument(String)
}
