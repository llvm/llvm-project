@_spi(Private) import B
public class FromA {
  public init() {}
  @_spi(Private) public var a : Int { get { return 42 } }
  @_spi(Private) public var b : FromB { get { return FromB() } }
}
