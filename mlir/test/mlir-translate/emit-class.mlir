// RUN: mlir-translate --mlir-to-cpp --emit-class=true --class-name=MyAdder --field-name-attribute=tf_saved_model.index_path /tmp/model_emitc.mlir | FileCheck %s --check-prefix=ADDER_TEST

// ADDER_TEST-LABEL: class MyAdder final {
// ADDER_TEST-NEXT: public:
// ADDER_TEST-DAG: float v1[1];
// ADDER_TEST-DAG: float v2[1];
// ADDER_TEST-DAG: float v3[1];      
// ADDER_TEST-NEXT: std::map<std::string, char*> _buffer_map {{ "another_feature", reinterpret_cast<char*>(v1) },{ "some_feature", reinterpret_cast<char*>(v2) },{ "output_0", reinterpret_cast<char*>(v3) }};
// ADDER_TEST-NEXT: char* getBufferForName(const std::string& name) const {
// ADDER_TEST-NEXT:     auto it = _buffer_map.find(name);
// ADDER_TEST-NEXT:     return (it == _buffer_map.end()) ? nullptr : it->second;
// ADDER_TEST-NEXT: }
// ADDER_TEST-NEXT: void main() {
// ADDER_TEST-NEXT:     size_t v4 = 0;
// ADDER_TEST-NEXT:     float v5 = v2[v4];
// ADDER_TEST-NEXT:     float v6 = v1[v4];
// ADDER_TEST-NEXT:     float v7 = v5 + v6;
// ADDER_TEST-NEXT:     v3[v4] = v7;
// ADDER_TEST-NEXT:     return;
// ADDER_TEST-NEXT: }
// ADDER_TEST-NEXT: };

// ---
// RUN: mlir-translate --mlir-to-cpp --emit-class=true --class-name=MyMultiOutput --field-name-attribute=tf_saved_model.index_path /tmp/model_multi_out_emitc.mlir | FileCheck %s --check-prefix=MULTI_OUT

// MULTI_OUT-LABEL: class MyMultiOutput final {
// MULTI_OUT-NEXT: public:
// MULTI_OUT-DAG: float v1[1];
// MULTI_OUT-DAG: float v2[1];
// MULTI_OUT-DAG: float v3[1];
// MULTI_OUT-DAG: float v4[1];
// MULTI_OUT: std::map<std::string, char*> _buffer_map {{ "b", reinterpret_cast<char*>(v1) },{ "a", reinterpret_cast<char*>(v2) },{ "output_1", reinterpret_cast<char*>(v3) },{ "output_0", reinterpret_cast<char*>(v4) }, };
// MULTI_OUT-NEXT: char* getBufferForName(const std::string& name) const {
// MULTI_OUT-NEXT:     auto it = _buffer_map.find(name);
// MULTI_OUT-NEXT:     return (it == _buffer_map.end()) ? nullptr : it->second;
// MULTI_OUT-NEXT: }
// MULTI_OUT-NEXT: void main() {
// MULTI_OUT-NEXT:     size_t v5 = 0;
// MULTI_OUT-NEXT:     float v6 = v2[v5];
// MULTI_OUT-NEXT:     float v7 = v1[v5];
// MULTI_OUT-NEXT:     float v8 = v6 + v7;
// MULTI_OUT-NEXT:     v4[v5] = v8;
// MULTI_OUT-NEXT:     float v9 = v2[v5];
// MULTI_OUT-NEXT:     float v10 = v1[v5];
// MULTI_OUT-NEXT:     float v11 = v9 - v10;
// MULTI_OUT-NEXT:     v3[v5] = v11;
// MULTI_OUT-NEXT:     return;
// MULTI_OUT-NEXT: }
// MULTI_OUT-NEXT: };
