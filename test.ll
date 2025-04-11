ByteAddressBuffer Buffer0: register(t0);
RWBuffer<float> Buf : register(u0);

[numthreads(4,1,1)]
void main() {
  Buf[0] = 10;
}
