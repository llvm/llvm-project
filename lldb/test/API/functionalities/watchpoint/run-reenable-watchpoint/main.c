static int arr[] = {1, 2, 0, 3, 4, 0x55555555};
int main() {
  arr[0]++; // break here
  arr[0]++;
  return arr[4];
}
