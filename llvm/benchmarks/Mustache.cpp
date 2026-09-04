#include "llvm/Support/Mustache.h"
#include "benchmark/benchmark.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

// A large, raw string with many characters that require HTML escaping.
static const std::string LongHtmlString = [] {
  std::string S;
  S.reserve(500000);
  for (int Idx = 0; Idx < 50000; ++Idx) {
    S += "<script>alert('xss');</script>";
  }
  return S;
}();

// A deep AND wide JSON object for testing traversal.
static const llvm::json::Value DeepJsonData = [] {
  llvm::json::Value Root = llvm::json::Object();
  llvm::json::Object *Current = Root.getAsObject();
  for (int i = 0; i < 50; ++i) { // 50 levels deep
    for (int j = 0; j < 100; ++j) {
      (*Current)["sibling_" + std::to_string(j)] = llvm::json::Value("noise");
    }
    std::string Key = "level_" + std::to_string(i);
    (*Current)[Key] = llvm::json::Object();
    Current = (*Current)[Key].getAsObject();
  }
  (*Current)["final_value"] = llvm::json::Value("Success!");

  llvm::json::Array Arr;
  for (int i = 0; i < 5000; ++i) { // 5,000 iterations
    Arr.push_back(llvm::json::Value(i));
  }

  llvm::json::Object NewRoot;
  NewRoot["deep_data"] = std::move(Root);
  NewRoot["loop_array"] = std::move(Arr);
  return llvm::json::Value(std::move(NewRoot));
}();

// A huge array for testing iteration performance.
static const llvm::json::Value HugeArrayData = [] {
  llvm::json::Array Arr;
  for (int i = 0; i < 100000; ++i) { // 100,000 array items
    Arr.push_back(llvm::json::Object(
        {{"id", llvm::json::Value(static_cast<long long>(i))},
         {"is_even", llvm::json::Value(i % 2 == 0)},
         {"data", llvm::json::Value("Item data for " + std::to_string(i))}}));
  }
  return llvm::json::Object({{"items", std::move(Arr)}});
}();

// The main template that includes a partial within a loop.
static const std::string ComplexPartialTemplate =
    "Header\n"
    "{{#items}}{{> item_partial}}{{/items}}\n"
    "Footer";

// The partial template is now more complex, rendering multiple fields and a
// conditional section.
static const std::string ItemPartialTemplate =
    "<div class=\"item\" id=\"{{id}}\">\n"
    " <p>{{data}}</p>\n"
    " {{#is_even}}<span>(Even)</span>{{/is_even}}\n"
    "</div>\n";

// A single large string to stress the output buffer.
static const llvm::json::Value LargeOutputData = llvm::json::Object({
    {"long_string",
     llvm::json::Value(std::string(1024 * 1024, 'A'))} // 1MB string
});

// --- Static Data (Templates) ---

static const std::string BulkEscapingTemplate = "{{content}}";
static const std::string BulkUnescapedTemplate = "{{{content}}}";
static const std::string BulkUnescapedAmpersandTemplate = "{{& content}}";

static const std::string DeepTraversalTemplate = [] {
  std::string LongKey =
      "deep_data.level_0.level_1.level_2.level_3.level_4.level_5."
      "level_6.level_7.level_8.level_9."
      "level_10.level_11.level_12.level_13.level_14.level_"
      "15.level_16.level_17.level_18.level_19."
      "level_20.level_21.level_22.level_23.level_24.level_"
      "25.level_26.level_27.level_28.level_29."
      "level_30.level_31.level_32.level_33.level_34.level_"
      "35.level_36.level_37.level_38.level_39."
      "level_40.level_41.level_42.level_43.level_44.level_"
      "45.level_46.level_47.level_48.level_49.final_value";
  return "{{#loop_array}}{{" + LongKey + "}}{{/loop_array}}";
}();

static const std::string DeeplyNestedRenderingTemplate = [] {
  std::string NestedTemplate = "{{#deep_data}}";
  for (int i = 0; i < 50; ++i) {
    NestedTemplate += "{{#level_" + std::to_string(i) + "}}";
  }
  NestedTemplate += "{{final_value}}";
  for (int i = 49; i >= 0; --i) {
    NestedTemplate += "{{/level_" + std::to_string(i) + "}}";
  }
  NestedTemplate += "{{/deep_data}}";
  return NestedTemplate;
}();

static const std::string HugeArrayIterationTemplate =
    "{{#items}}ID: {{id}}.{{/items}}";

static const std::string ComplexTemplateParsingTemplate = [] {
  std::string LargeTemplate;
  LargeTemplate.reserve(100000);
  for (int i = 0; i < 1000; ++i) {
    LargeTemplate += "{{var_" + std::to_string(i) +
                     "}}"
                     "{{#section_" +
                     std::to_string(i) + "}}Content{{/section_" +
                     std::to_string(i) +
                     "}}"
                     "{{!comment_" +
                     std::to_string(i) +
                     "}}"
                     "{{=<% %>=}}"
                     "<%var_tag_changed_to_percent_sign_" +
                     std::to_string(i) +
                     "%>"
                     "<%={{ }}=%>"
                     "{{^inverted_" +
                     std::to_string(i) + "}}Not Present{{/inverted_" +
                     std::to_string(i) + "}}";
  }
  return LargeTemplate;
}();

static const std::string SmallTemplateParsingTemplate =
    "{{level_0.sibling_99}}\n"
    "{{level_0.level_1.level_2.level_3.level_4.level_5.sibling_50}}\n"
    "{{level_0.level_1.level_2.level_3.level_4.level_5."
    "level_6.level_7.level_8.level_9."
    "level_10.level_11.level_12.level_13.level_14.level_"
    "15.level_16.level_17.level_18.level_19."
    "level_20.level_21.level_22.level_23.level_24.level_"
    "25.level_26.level_27.level_28.level_29."
    "level_30.level_31.level_32.level_33.level_34.level_"
    "35.level_36.level_37.level_38.level_39."
    "level_40.level_41.level_42.level_43.level_44.level_"
    "45.level_46.level_47.level_48.level_49.final_value}}\n";

static const std::string LargeOutputStringTemplate = "{{long_string}}";

// Tests the performance of rendering a large string with various escaping
// syntaxes.
static void BM_Mustache_StringRendering(benchmark::State &state,
                                        const std::string &TplStr) {
  llvm::BumpPtrAllocator Allocator;
  llvm::StringSaver Saver(Allocator);
  llvm::mustache::MustacheContext Ctx(Allocator, Saver);

  llvm::mustache::Template Tpl(TplStr, Ctx);
  llvm::json::Value Data =
      llvm::json::Object({{"content", llvm::json::Value(LongHtmlString)}});
  for (auto _ : state) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Tpl.render(Data, OS);
    benchmark::DoNotOptimize(Result);
  }
}
BENCHMARK_CAPTURE(BM_Mustache_StringRendering, Escaped, BulkEscapingTemplate);
BENCHMARK_CAPTURE(BM_Mustache_StringRendering, Unescaped_Triple,
                  BulkUnescapedTemplate);
BENCHMARK_CAPTURE(BM_Mustache_StringRendering, Unescaped_Ampersand,
                  BulkUnescapedAmpersandTemplate);

// Tests the "hot render" cost of repeatedly traversing a deep and wide
// JSON object.
static void BM_Mustache_DeepTraversal(benchmark::State &state) {
  llvm::BumpPtrAllocator Allocator;
  llvm::StringSaver Saver(Allocator);
  llvm::mustache::MustacheContext Ctx(Allocator, Saver);

  llvm::mustache::Template Tpl(DeepTraversalTemplate, Ctx);
  for (auto _ : state) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Tpl.render(DeepJsonData, OS);
    benchmark::DoNotOptimize(Result);
  }
}
BENCHMARK(BM_Mustache_DeepTraversal);

// Tests the "hot render" cost of pushing and popping a deep context stack.
static void BM_Mustache_DeeplyNestedRendering(benchmark::State &state) {

  llvm::BumpPtrAllocator Allocator;
  llvm::StringSaver Saver(Allocator);
  llvm::mustache::MustacheContext Ctx(Allocator, Saver);

  llvm::mustache::Template Tpl(DeeplyNestedRenderingTemplate, Ctx);
  for (auto _ : state) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Tpl.render(DeepJsonData, OS);
    benchmark::DoNotOptimize(Result);
  }
}
BENCHMARK(BM_Mustache_DeeplyNestedRendering);

// Tests the performance of the loop logic when iterating over a huge number of
// items.
static void BM_Mustache_HugeArrayIteration(benchmark::State &state) {
  llvm::BumpPtrAllocator Allocator;
  llvm::StringSaver Saver(Allocator);
  llvm::mustache::MustacheContext Ctx(Allocator, Saver);

  llvm::mustache::Template Tpl(HugeArrayIterationTemplate, Ctx);
  for (auto _ : state) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Tpl.render(HugeArrayData, OS);
    benchmark::DoNotOptimize(Result);
  }
}
BENCHMARK(BM_Mustache_HugeArrayIteration);

// Tests the performance of the parser on a large, "wide" template.
static void BM_Mustache_ComplexTemplateParsing(benchmark::State &state) {
  llvm::BumpPtrAllocator Allocator;
  llvm::StringSaver Saver(Allocator);
  llvm::mustache::MustacheContext Ctx(Allocator, Saver);

  for (auto _ : state) {
    llvm::mustache::Template Tpl(ComplexTemplateParsingTemplate, Ctx);
    benchmark::DoNotOptimize(Tpl);
  }
}
BENCHMARK(BM_Mustache_ComplexTemplateParsing);

// Tests the performance of the parser on a small, "deep" template.
static void BM_Mustache_SmallTemplateParsing(benchmark::State &state) {
  llvm::BumpPtrAllocator Allocator;
  llvm::StringSaver Saver(Allocator);
  llvm::mustache::MustacheContext Ctx(Allocator, Saver);

  for (auto _ : state) {
    llvm::mustache::Template Tpl(SmallTemplateParsingTemplate, Ctx);
    benchmark::DoNotOptimize(Tpl);
  }
}
BENCHMARK(BM_Mustache_SmallTemplateParsing);

// Tests the performance of rendering a template that includes a partial.
static void BM_Mustache_PartialsRendering(benchmark::State &state) {
  llvm::BumpPtrAllocator Allocator;
  llvm::StringSaver Saver(Allocator);
  llvm::mustache::MustacheContext Ctx(Allocator, Saver);

  llvm::mustache::Template Tpl(ComplexPartialTemplate, Ctx);
  Tpl.registerPartial("item_partial", ItemPartialTemplate);
  llvm::json::Value Data = HugeArrayData;

  for (auto _ : state) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Tpl.render(Data, OS);
    benchmark::DoNotOptimize(Result);
  }
}
BENCHMARK(BM_Mustache_PartialsRendering);

// Tests the performance of the underlying buffer management when generating a
// very large output.
static void BM_Mustache_LargeOutputString(benchmark::State &state) {
  llvm::BumpPtrAllocator Allocator;
  llvm::StringSaver Saver(Allocator);
  llvm::mustache::MustacheContext Ctx(Allocator, Saver);

  llvm::mustache::Template Tpl(LargeOutputStringTemplate, Ctx);
  for (auto _ : state) {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    Tpl.render(LargeOutputData, OS);
    benchmark::DoNotOptimize(Result);
  }
}
BENCHMARK(BM_Mustache_LargeOutputString);

BENCHMARK_MAIN();
