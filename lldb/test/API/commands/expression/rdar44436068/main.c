int main(void)
{
    __int128_t n = 1;
    n = n + n;
    return n; //%self.expect("p n", substrs=['(__int128_t) $0 = 2'])
              //%self.expect("p n + 6", substrs=['(__int128) $1 = 8'])
              //%self.expect("p n + n", substrs=['(__int128) $2 = 4'])
}
