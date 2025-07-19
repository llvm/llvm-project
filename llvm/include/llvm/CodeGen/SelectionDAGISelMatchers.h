namespace llvm {
namespace SDPatternMatch {

// 1. 定義 SelectCC_match
template <typename LTy, typename RTy, typename TTy, typename FTy, typename CCTy>
struct SelectCC_match {
  const LTy &L;
  const RTy &R;
  const TTy &T;
  const FTy &F;
  const CCTy &CC;

  SelectCC_match(const LTy &l, const RTy &r,
                 const TTy &t, const FTy &f,
                 const CCTy &cc)
      : L(l), R(r), T(t), F(f), CC(cc) {}

  template <typename OpTy>
  bool match(OpTy V) const {
    if (V.getOpcode() != ISD::SELECT_CC)
      return false;

    return L.match(V.getOperand(0)) &&
           R.match(V.getOperand(1)) &&
           T.match(V.getOperand(2)) &&
           F.match(V.getOperand(3)) &&
           CC.match(cast<CondCodeSDNode>(V.getOperand(4))->get());
  }
};

// 2. 定義 m_SelectCC
template <typename LTy, typename RTy, typename TTy, typename FTy, typename CCTy>
inline SelectCC_match<LTy, RTy, TTy, FTy, CCTy>
m_SelectCC(const LTy &L, const RTy &R,
           const TTy &T, const FTy &F,
           const CCTy &CC) {
  return SelectCC_match<LTy, RTy, TTy, FTy, CCTy>(L, R, T, F, CC);
}

// 3. 定義 m_SelectCCLike
template <typename LTy, typename RTy, typename TTy, typename FTy, typename CCTy>
inline auto m_SelectCCLike(const LTy &L, const RTy &R,
                           const TTy &T, const FTy &F,
                           const CCTy &CC) {
  return m_AnyOf(
    m_Select(m_SetCC(L, R, CC), T, F),
    m_SelectCC(L, R, T, F, CC)
  );
}

} // namespace SDPatternMatch
} // namespace llvm

