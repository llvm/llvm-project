define float @declared_as_nonan(float %arg) {
  %add = fadd float %arg, 1.0
  ret float %add
}
