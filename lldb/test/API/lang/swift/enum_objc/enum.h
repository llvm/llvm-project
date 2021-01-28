enum __attribute__((enum_extensibility(closed))) ComparisonResult : long {
    OrderedAscending = -1L,
    OrderedSame,
    OrderedDescending
};

enum ComparisonResult getReturn(long x) {
  switch (x) {
    case 0:
      return OrderedSame;
    case -1:
      return OrderedAscending;
    case 1:
    default:
      return OrderedDescending;
  }
}
