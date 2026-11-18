class Model {
public:
  int64_t fieldName0[33];
  const std::map<std::string, char *> reflectionMap = {
      {"mask", reinterpret_cast<char *>(&fieldName0)}};
  char *getBufferForName(std::string v1) {
    char *v2 = reflectionMap.at(v1);
    return v2;
  }
  int64_t operator()() {
    size_t v1 = 0;
    size_t v2 = 1;
    size_t v3 = 33;
    size_t v4 = 0;
    int32_t v5 = -2147483648;
    int64_t v6 = 0;
    int32_t v7[33];
    for (size_t i8 = v4; i8 < v3; i8 += v2) {
      int64_t v9 = fieldName0[i8];
      uint64_t v10 = (uint64_t)v9;
      uint32_t v11 = (uint32_t)v10;
      int32_t v12 = (int32_t)v11;
      v7[i8] = v12;
    }
    int64_t v13;
    int64_t *v14 = &v13;
    v14[v1] = v6;
    int32_t v15;
    int32_t *v16 = &v15;
    v16[v1] = v5;
    for (size_t i17 = v4; i17 < v3; i17 += v2) {
      int32_t v18 = v7[i17];
      int64_t v19 = v14[v1];
      int32_t v20 = v16[v1];
      ptrdiff_t v21 = (ptrdiff_t)i17;
      int64_t v22 = (int64_t)v21;
      bool v23 = v18 > v20;
      bool v24 = v18 > v20;
      int32_t v25 = v24 ? v18 : v20;
      int64_t v26 = v23 ? v22 : v19;
      v14[v1] = v26;
      v16[v1] = v25;
    }
    int64_t v27 = v14[v1];
    return v27;
  }
};
